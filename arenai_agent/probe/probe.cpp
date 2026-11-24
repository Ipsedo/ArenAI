#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <thread>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <arenai_model/constants.h>
#include <arenai_model/shell_economy.h>
#include <arenai_view/backend.h>

#include "./agents/ppo_liquid/liquid_hidden_state.h"
#include "./agents/sac/sac_discrete.h"
#include "./agents/torch_types.h"
#include "./core/train_environment.h"
#include "./distributions/bernoulli.h"
#include "./distributions/gaussian_tanh.h"
#include "./distributions/spatial_aim_law.h"
#include "./networks/actor.h"
#include "./networks/q_function.h"
#include "./networks/recurrent/liquid_actor.h"
#include "./networks/recurrent/liquid_critic.h"
#include "./networks/vision.h"
#include "./networks_utils/torch_loader.h"

using namespace arenai;
using namespace arenai::agent;

namespace {
    // dir_x, dir_y lead the continuous actions, the aim ones follow
    constexpr int NB_DIRECTION_ACTIONS = 2;
}// namespace

int main(const int argc, char **argv) {
    argparse::ArgumentParser parser("arenai probe");

    parser.add_argument("--run_folder").required();
    parser.add_argument("--save").required();
    parser.add_argument("--resources_folder").required();
    parser.add_argument("--output").required();
    parser.add_argument("--nb_episodes").scan<'i', int>().default_value(4);
    parser.add_argument("--spawn_side").scan<'g', float>().default_value(550.f);
    parser.add_argument("--shells")
        .scan<'i', int>()
        .default_value(model::GAME_SHELL_ECONOMY.initial_shells);
    parser.add_argument("--shells_per_hit")
        .scan<'i', int>()
        .default_value(model::GAME_SHELL_ECONOMY.shells_per_hit);
    parser.add_argument("--max_shells")
        .scan<'i', int>()
        .default_value(model::GAME_SHELL_ECONOMY.max_shells);
    parser.add_argument("--cuda").default_value(false).implicit_value(true);
    parser.add_argument("--deterministic").default_value(false).implicit_value(true);
    parser.add_argument("--features").default_value(std::string());
    parser.add_argument("--frames").default_value(std::string());
    parser.add_argument("--sightings").default_value(std::string());
    parser.add_argument("--presence").default_value(std::string());
    parser.add_argument("--frames_every").scan<'i', int>().default_value(5);
    parser.add_argument("--zoom")
        .default_value(std::string("policy"))
        .choices("policy", "on", "off");
    parser.add_argument("--oracle").default_value(false).implicit_value(true);
    parser.add_argument("--oracle_fov").scan<'g', float>().default_value(360.f);
    parser.add_argument("--oracle_policy_yaw").default_value(false).implicit_value(true);
    parser.add_argument("--oracle_policy_pitch").default_value(false).implicit_value(true);
    parser.add_argument("--oracle_policy_fire").default_value(false).implicit_value(true);
    parser.add_argument("--oracle_tolerance").scan<'g', float>().default_value(1.f);
    parser.add_argument("--oracle_target")
        .default_value(std::string("geometry"))
        .choices("geometry", "network");
    parser.add_argument("--vision_num_threads")
        .scan<'i', int>()
        .default_value(static_cast<int>(std::thread::hardware_concurrency()));

    parser.parse_args(argc, argv);

    const std::filesystem::path run_folder(parser.get<std::string>("--run_folder"));
    const std::filesystem::path save_folder = run_folder / parser.get<std::string>("--save");
    const std::filesystem::path resources_folder(parser.get<std::string>("--resources_folder"));
    const std::filesystem::path output(parser.get<std::string>("--output"));
    const int nb_episodes = parser.get<int>("--nb_episodes");
    const float spawn_side = parser.get<float>("--spawn_side");
    const model::ShellEconomy shell_economy{
        .initial_shells = parser.get<int>("--shells"),
        .shells_per_hit = parser.get<int>("--shells_per_hit"),
        .max_shells = parser.get<int>("--max_shells")};
    const bool deterministic = parser.get<bool>("--deterministic");
    const auto zoom_mode = parser.get<std::string>("--zoom");
    const bool oracle = parser.get<bool>("--oracle");
    const bool oracle_policy_yaw = parser.get<bool>("--oracle_policy_yaw");
    const bool oracle_policy_pitch = parser.get<bool>("--oracle_policy_pitch");
    const bool oracle_policy_fire = parser.get<bool>("--oracle_policy_fire");
    const float oracle_tolerance =
        parser.get<float>("--oracle_tolerance") * std::numbers::pi_v<float> / 180.f;
    const bool network_target = parser.get<std::string>("--oracle_target") == "network";
    const float oracle_half_yaw =
        std::min(parser.get<float>("--oracle_fov"), 360.f) * std::numbers::pi_v<float> / 360.f;

    const torch::Device device =
        parser.get<bool>("--cuda") ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    nlohmann::json config;
    std::ifstream(run_folder / "config.json") >> config;
    const auto &agent_config = config.at("agent");
    const auto &env_config = config.at("environment");

    const int vision_height = env_config.at("vision_height").get<int>();
    const int vision_width = env_config.at("vision_width").get<int>();
    const float wanted_frequency = env_config.at("wanted_frequency").get<float>();
    const int nb_tanks = env_config.at("nb_tanks").get<int>();
    const int action_repeat = env_config.at("action_repeat").get<int>();
    const int max_episode_steps = config.at("train").at("max_episode_steps").get<int>();

    // what the probe needs from an actor, whatever its family (ppo_liquid, sac or ppg)
    struct ProbeActorOutput {
        torch::Tensor mu, sigma, discrete;
        // empty for actors without an exposed feature layer (ppg)
        torch::Tensor features, vision_features;
        // [N, 1] critic state value in scaled-return units, empty without a state-value critic
        torch::Tensor value;
        // [N, rows, cols] actor presence logits, empty unless the oracle targets the network's pick
        torch::Tensor presence;
        // the aim point law of the PPO families, absent for SAC whose aim point is in mu
        std::optional<SpatialAimLaw> aim;
    };
    std::function<ProbeActorOutput(const torch::Tensor &, const torch::Tensor &)> act;
    std::function<void()> reset_hidden = [] {};
    // [N, 3 × nb_aim] ∂E[min Q], ∂E[Q1], ∂E[Q2] /∂a_aim at the mode, undefined without a critic
    std::function<torch::Tensor(const TorchState &, const torch::Tensor &, const torch::Tensor &)>
        canon_q_gradient = [](const TorchState &, const torch::Tensor &, const torch::Tensor &) {
            return torch::Tensor();
        };

    const auto vision_channels =
        agent_config.at("vision_channels").get<std::vector<std::tuple<int, int>>>();
    const auto group_norm_nums = agent_config.at("group_norm_nums").get<std::vector<int>>();
    const std::vector initial_discrete_probas{
        agent_config.at("initial_fire_proba").get<float>(),
        agent_config.at("initial_zoom_proba").get<float>()};

    const bool is_ppg = agent_config.contains("beta_clone");
    const bool is_sac = !is_ppg && !agent_config.contains("neuron_number");
    std::cout << "actor family : " << (is_ppg ? "ppg" : is_sac ? "sac" : "ppo_liquid") << std::endl;

    if (is_ppg) {
        const auto actor = std::make_shared<DistillationActor>(
            vision_height, vision_width, model::ENEMY_STATE_VECTOR_SIZE,
            model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION,
            agent_config.at("hidden_size_sensors").get<int>(),
            agent_config.at("actor_hidden_sizes").get<std::vector<int>>(), vision_channels,
            group_norm_nums, agent_config.at("initial_sigma").get<float>(),
            initial_discrete_probas);
        load_torch(save_folder, actor, "actor.pt");
        actor->to(device);
        actor->eval();

        act = [actor](const torch::Tensor &vision, const torch::Tensor &sensors) {
            auto out = actor->act(vision, sensors);
            return ProbeActorOutput{
                .mu = out.mu, .sigma = out.sigma, .discrete = out.discrete, .aim = out.aim};
        };
    } else if (is_sac) {
        const auto make_encoder = [&] {
            return std::make_shared<StateEncoder>(
                vision_height, vision_width, model::ENEMY_STATE_VECTOR_SIZE,
                agent_config.at("hidden_size_sensors").get<int>(), vision_channels,
                group_norm_nums);
        };
        const auto actor_encoder = make_encoder();
        const auto actor = std::make_shared<SacActor>(
            actor_encoder, model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION,
            agent_config.at("actor_hidden_sizes").get<std::vector<int>>(),
            agent_config.at("initial_sigma").get<float>(), initial_discrete_probas);
        load_torch(save_folder, actor, "actor.pt");
        actor->to(device);
        actor->eval();

        const auto critic = std::make_shared<TwinQNetwork>(
            make_encoder(), model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION,
            agent_config.at("critic_hidden_sizes").get<std::vector<int>>());
        load_torch(save_folder, critic, "critic.pt");
        critic->to(device);
        critic->eval();

        act = [actor, actor_encoder,
               critic](const torch::Tensor &vision, const torch::Tensor &sensors) {
            auto out = actor->act(vision, sensors);
            // features: encoding saved with the actor, vision_features: saved with the critic
            return ProbeActorOutput{
                .mu = out.mu,
                .sigma = out.sigma,
                .discrete = out.discrete,
                .features = actor_encoder->forward(vision, sensors),
                .vision_features = critic->encode(vision, sensors)};
        };

        canon_q_gradient = [critic, device](
                               const TorchState &state, const torch::Tensor &mu,
                               const torch::Tensor &discrete) {
            const auto vision = state.get_vision().to(device);
            const auto sensors = state.get_state().to(device);
            const auto mask =
                TorchState::discrete_action_mask(state.get_fire_mask(), discrete.size(-1))
                    .to(device);
            const auto action = gaussian_tanh_mode(mu.to(device)).detach().requires_grad_(true);
            const auto [q1, q2] = critic->q(vision, sensors, action);
            const auto joint = discrete_joint(discrete.to(device), mask);
            const auto value = (joint.probas * torch::min(q1, q2)).sum();
            const auto value_1 = (joint.probas * q1).sum();
            const auto value_2 = (joint.probas * q2).sum();
            const auto gradient_of = [&action](const torch::Tensor &v) {
                return torch::autograd::grad({v}, {action}, {}, true)[0].slice(
                    1, NB_DIRECTION_ACTIONS);
            };
            return torch::cat({gradient_of(value), gradient_of(value_1), gradient_of(value_2)}, 1)
                .cpu();
        };
    } else {
        const auto actor = std::make_shared<LiquidActor>(
            vision_height, vision_width, model::ENEMY_STATE_VECTOR_SIZE,
            model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION,
            agent_config.at("hidden_size_sensors").get<int>(), vision_channels, group_norm_nums,
            agent_config.at("neuron_number").get<int>(),
            agent_config.at("unfolding_steps").get<int>(), agent_config.at("delta_t").get<float>(),
            agent_config.at("initial_sigma").get<float>(), initial_discrete_probas);
        load_torch(save_folder, actor, "actor.pt");
        actor->to(device);
        actor->eval();

        const auto critic = std::make_shared<LiquidCritic>(
            vision_height, vision_width, model::ENEMY_STATE_VECTOR_SIZE,
            agent_config.at("hidden_size_sensors").get<int>(), vision_channels, group_norm_nums,
            agent_config.at("neuron_number").get<int>(),
            agent_config.at("unfolding_steps").get<int>(), agent_config.at("delta_t").get<float>());
        load_torch(save_folder, critic, "critic.pt");
        critic->to(device);
        critic->eval();

        const auto hidden_state = std::make_shared<LiquidHiddenState>(actor);
        const auto critic_x = std::make_shared<torch::Tensor>();
        reset_hidden = [hidden_state, critic_x] {
            hidden_state->reset();
            *critic_x = torch::Tensor();
        };
        act = [actor, hidden_state, critic, critic_x,
               network_target](const torch::Tensor &vision, const torch::Tensor &sensors) {
            const auto x_t = hidden_state->get(vision.size(0));
            auto out = actor->act(vision, sensors, x_t);
            hidden_state->set(out.next_x);

            if (!critic_x->defined())
                *critic_x = critic->initial_state(static_cast<int>(vision.size(0)));
            const auto [value, next_critic_x] = critic->value(vision, sensors, *critic_x);
            *critic_x = next_critic_x;

            return ProbeActorOutput{
                .mu = out.mu,
                .sigma = out.sigma,
                .discrete = out.discrete,
                .features = out.features,
                .vision_features = out.vision_features,
                .value = value,
                .presence = network_target ? actor->presence_logits(vision) : torch::Tensor(),
                .aim = out.aim};
        };
    }

    auto graphics_backend = view::make_vulkan_backend();
    std::cout << "Vulkan device : " << graphics_backend->renderer_info() << std::endl;

    const auto env = std::make_unique<TrainTankEnvironment>(
        std::move(graphics_backend), nb_tanks, resources_folder, wanted_frequency,
        max_episode_steps, vision_height, vision_width, parser.get<int>("--vision_num_threads"),
        action_repeat);

    std::ofstream csv(output);
    const auto &action_names = TorchAction::continuous_action_names();
    const std::vector aim_names(action_names.begin() + NB_DIRECTION_ACTIONS, action_names.end());

    csv << "episode;step;tank;distance;yaw;pitch;best_angle;best_distance;can_fire;shells;cooldown;"
           "p_fire;p_zoom;fire";
    for (const auto &prefix: {";mu_", ";sigma_"})
        for (int a = 0; a < NB_DIRECTION_ACTIONS; a++) csv << prefix << action_names[a];
    for (const auto &name: action_names) csv << ";act_" << name;
    csv << ";aim_cell;aim_log_prob;aim_cell_entropy";
    for (const auto &prefix: {";dq_", ";dq1_", ";dq2_"})
        for (const auto &name: aim_names) csv << prefix << name;
    csv << ";turret_yaw;canon_elevation;canon_relative_elevation;chassis_pitch;chassis_roll;"
           "value;nearest_sight_fraction\n";

    std::ofstream shots_csv(output.parent_path() / (output.stem().string() + "_shots.csv"));
    shots_csv
        << "episode;tank;fire_step;land_step;fire_best_angle;fire_best_distance;fire_distance;"
           "fire_yaw;fire_pitch;shells_at_fire;group_size;group_hits;group_miss\n";

    std::ofstream rewards_csv(output.parent_path() / (output.stem().string() + "_rewards.csv"));
    rewards_csv << "episode;step;tank;reward;done;truncated\n";

    const auto sightings_path = parser.get<std::string>("--sightings");
    std::ofstream sightings_out;
    if (!sightings_path.empty()) sightings_out.open(sightings_path, std::ios::binary);

    const auto presence_path = parser.get<std::string>("--presence");
    std::ofstream presence_out;
    if (!presence_path.empty()) presence_out.open(presence_path, std::ios::binary);
    const auto [presence_rows, presence_cols] =
        presence_grid_size(vision_height, vision_width, static_cast<int>(vision_channels.size()));

    const auto features_path = parser.get<std::string>("--features");
    std::ofstream features_out;
    if (!features_path.empty()) features_out.open(features_path, std::ios::binary);

    const auto frames_path = parser.get<std::string>("--frames");
    const int frames_every = parser.get<int>("--frames_every");
    std::ofstream frames_out;
    if (!frames_path.empty()) frames_out.open(frames_path, std::ios::binary);
    constexpr float frames_half_yaw = 39.5f * std::numbers::pi_v<float> / 180.f;

    struct PendingShot {
        int step;
        AimGeometry geometry;
        float shells;
    };

    constexpr float to_degrees = 180.f / std::numbers::pi_v<float>;

    for (int episode = 0; episode < nb_episodes; episode++) {
        reset_hidden();

        auto state = TorchState(
            env->reset(spawn_side, spawn_side, shell_economy), vision_height, vision_width);

        int step = 0;
        bool is_done = false;
        float episode_kills = 0.f;
        int episode_hits = 0, episode_fired = 0;
        long network_designations = 0, network_agreements = 0;
        std::vector<std::deque<PendingShot>> pending(nb_tanks);

        while (!is_done) {
            torch::Tensor continuous_action, discrete_action, mu, sigma, discrete, features,
                vision_features, value, presence_probas, aim_cell, aim_log_prob, aim_cell_entropy;
            {
                torch::NoGradGuard guard;

                auto out = act(state.get_vision().to(device), state.get_state().to(device));

                mu = out.mu.cpu();
                sigma = out.sigma.cpu();
                discrete = out.discrete.cpu();
                if (out.value.defined()) value = out.value.cpu().reshape({-1});
                if (out.presence.defined()) presence_probas = torch::sigmoid(out.presence).cpu();
                if (out.features.defined()) {
                    features = out.features.cpu().contiguous();
                    vision_features = out.vision_features.cpu().contiguous();
                }

                if (deterministic) {
                    continuous_action = gaussian_tanh_mode(mu);
                    discrete_action = bernoulli_max_action(discrete);
                } else {
                    continuous_action = gaussian_tanh_sample(mu, sigma).first;
                    discrete_action = bernoulli_sample(discrete);
                }

                // the PPO families close the command with a point drawn from their aim law
                if (out.aim.has_value()) {
                    const auto draw = deterministic ? out.aim->mode() : out.aim->sample();
                    continuous_action =
                        torch::cat({continuous_action, out.aim->point(draw).cpu()}, -1);
                    aim_cell = draw.cell.cpu();
                    aim_log_prob = out.aim->log_prob(draw).cpu().reshape({-1});
                    aim_cell_entropy = out.aim->cell_entropy().cpu().reshape({-1});
                }

                // discrete actions : fire, zoom
                if (zoom_mode != "policy")
                    discrete_action.select(1, 1).fill_(zoom_mode == "on" ? 1.f : 0.f);
            }

            const auto q_gradient = canon_q_gradient(state, mu, discrete);

            const auto geometry = env->aim_geometry(oracle_half_yaw);
            const auto presence = presence_out.is_open()
                                      ? env->presence_maps(presence_rows, presence_cols)
                                      : torch::Tensor();
            const auto sightings =
                sightings_out.is_open() || oracle ? env->enemy_sightings(
                    static_cast<float>(vision_width) / static_cast<float>(vision_height))
                                                  : std::vector<std::vector<EnemySighting>>();
            const bool write_frames = frames_out.is_open() && step % frames_every == 0;
            const auto frame_geometry =
                write_frames ? env->aim_geometry(frames_half_yaw) : std::vector<AimGeometry>();
            const auto frames =
                write_frames ? state.get_vision().cpu().contiguous() : torch::Tensor();

            // oracle: aims at the image point of the target and fires once the canon is aligned
            if (oracle) {
                auto cont = continuous_action.accessor<float, 2>();
                auto disc = discrete_action.accessor<float, 2>();
                for (int i = 0; i < nb_tanks; i++) {
                    if (!oracle_policy_fire) disc[i][0] = 0.f;
                    if (!geometry[i].alive) continue;

                    // geometry target: the sighting of the nearest enemy in view
                    float target_gap = std::numeric_limits<float>::max();
                    const auto gap_of = [&](const EnemySighting &sighting) {
                        if (network_target) {
                            const auto probas = presence_probas[i].reshape({-1});
                            const auto best = probas.argmax().item<int64_t>();
                            if (probas[best].item<float>() < 0.5f)
                                return std::numeric_limits<float>::max();
                            const float row = static_cast<float>(best / presence_cols) + 0.5f;
                            const float col = static_cast<float>(best % presence_cols) + 0.5f;
                            const float gap_row =
                                (1.f - sighting.ndc_y) / 2.f * static_cast<float>(presence_rows)
                                - row;
                            const float gap_col =
                                (sighting.ndc_x + 1.f) / 2.f * static_cast<float>(presence_cols)
                                - col;
                            return gap_row * gap_row + gap_col * gap_col;
                        }
                        if (geometry[i].view_distance < 0.f)
                            return std::numeric_limits<float>::max();
                        return std::abs(sighting.canon_yaw - geometry[i].view_yaw)
                               + std::abs(sighting.canon_pitch - geometry[i].view_pitch);
                    };

                    const EnemySighting *target = nullptr;
                    for (const auto &sighting: sightings[i]) {
                        if (sighting.clip_w <= 0.f || std::abs(sighting.ndc_x) >= 1.f
                            || std::abs(sighting.ndc_y) >= 1.f)
                            continue;
                        if (const float gap = gap_of(sighting); gap < target_gap) {
                            target_gap = gap;
                            target = &sighting;
                        }
                    }
                    if (target == nullptr) continue;

                    if (network_target) {
                        network_designations++;
                        if (geometry[i].view_distance >= 0.f
                            && std::abs(target->canon_yaw - geometry[i].view_yaw) < 0.01f
                            && std::abs(target->canon_pitch - geometry[i].view_pitch) < 0.01f)
                            network_agreements++;
                    }

                    // the image point carries no drop nor parallax correction
                    if (!oracle_policy_yaw) cont[i][NB_DIRECTION_ACTIONS] = target->ndc_x;
                    if (!oracle_policy_pitch) cont[i][NB_DIRECTION_ACTIONS + 1] = target->ndc_y;

                    const float drop = std::atan(
                        model::GRAVITY * target->distance
                        / (2.f * model::SHELL_SPEED * model::SHELL_SPEED));
                    if (!oracle_policy_fire)
                        disc[i][0] =
                            std::abs(target->canon_yaw) < oracle_tolerance
                                    && std::abs(target->canon_pitch + drop) < oracle_tolerance
                                ? 1.f
                                : 0.f;
                }
            }

            const auto mu_acc = mu.accessor<float, 2>();
            const auto sigma_acc = sigma.accessor<float, 2>();
            const auto discrete_acc = discrete.accessor<float, 2>();
            const auto cont_acc = continuous_action.accessor<float, 2>();
            const auto disc_acc = discrete_action.accessor<float, 2>();
            const auto shells_ratio = state.get_remaining_shells_ratio();
            const auto cooldown_ratio = state.get_cooldown_ratio();
            const auto mask_acc = state.get_fire_mask().accessor<float, 2>();

            for (int i = 0; i < nb_tanks; i++) {
                if (!geometry[i].alive) continue;

                csv << episode << ';' << step << ';' << i << ';' << geometry[i].distance << ';'
                    << geometry[i].yaw * to_degrees << ';' << geometry[i].pitch * to_degrees << ';'
                    << geometry[i].best_angle * to_degrees << ';' << geometry[i].best_distance
                    << ';' << mask_acc[i][0] << ';' << shells_ratio[i] << ';' << cooldown_ratio[i]
                    << ';' << discrete_acc[i][0] << ';' << discrete_acc[i][1] << ';'
                    << disc_acc[i][0];
                for (const auto &acc: {mu_acc, sigma_acc})
                    for (int a = 0; a < NB_DIRECTION_ACTIONS; a++) csv << ';' << acc[i][a];
                for (int a = 0; a < model::ENEMY_NB_CONTINUOUS_ACTION; a++)
                    csv << ';' << cont_acc[i][a];
                if (aim_cell.defined())
                    csv << ';' << aim_cell[i].item<int64_t>() << ';'
                        << aim_log_prob[i].item<float>() << ';'
                        << aim_cell_entropy[i].item<float>();
                else csv << ";-1;0;0";
                for (int a = 0; a < 3 * static_cast<int>(aim_names.size()); a++)
                    csv << ';' << (q_gradient.defined() ? q_gradient[i][a].item<float>() : 0.f);
                csv << ';' << geometry[i].turret_yaw * to_degrees << ';'
                    << geometry[i].canon_elevation * to_degrees << ';'
                    << geometry[i].canon_relative_elevation * to_degrees << ';'
                    << geometry[i].chassis_pitch * to_degrees << ';'
                    << geometry[i].chassis_roll * to_degrees << ';'
                    << (value.defined() ? value[i].item<float>() : 0.f) << ';'
                    << geometry[i].nearest_sight_fraction << '\n';

                if (presence_out.is_open()) {
                    const float header[3] = {
                        static_cast<float>(episode), static_cast<float>(step),
                        static_cast<float>(i)};
                    presence_out.write(reinterpret_cast<const char *>(header), sizeof(header));
                    presence_out.write(
                        reinterpret_cast<const char *>(presence[i].contiguous().data_ptr<float>()),
                        presence_rows * presence_cols * sizeof(float));
                }

                if (sightings_out.is_open()) {
                    std::vector<float> record{
                        static_cast<float>(episode), static_cast<float>(step),
                        static_cast<float>(i), static_cast<float>(sightings[i].size())};
                    for (const auto &sighting: sightings[i])
                        record.insert(
                            record.end(), {sighting.ndc_x, sighting.ndc_y, sighting.clip_w,
                                           sighting.distance, sighting.sight_fraction});
                    record.resize(4 + 5 * (nb_tanks - 1), 0.f);
                    sightings_out.write(
                        reinterpret_cast<const char *>(record.data()),
                        static_cast<std::streamsize>(record.size() * sizeof(float)));
                }

                if (features_out.is_open() && features.defined()) {
                    const float header[3] = {
                        static_cast<float>(episode), static_cast<float>(step),
                        static_cast<float>(i)};
                    features_out.write(reinterpret_cast<const char *>(header), sizeof(header));
                    features_out.write(
                        reinterpret_cast<const char *>(features[i].data_ptr<float>()),
                        features.size(1) * sizeof(float));
                    features_out.write(
                        reinterpret_cast<const char *>(vision_features[i].data_ptr<float>()),
                        vision_features.size(1) * sizeof(float));
                }

                if (write_frames) {
                    const float header[10] = {
                        static_cast<float>(episode),
                        static_cast<float>(step),
                        static_cast<float>(i),
                        frame_geometry[i].view_distance,
                        frame_geometry[i].view_yaw * to_degrees,
                        frame_geometry[i].view_pitch * to_degrees,
                        frame_geometry[i].best_distance,
                        frame_geometry[i].best_yaw * to_degrees,
                        frame_geometry[i].best_pitch * to_degrees,
                        static_cast<float>(frame_geometry[i].view_count)};
                    frames_out.write(reinterpret_cast<const char *>(header), sizeof(header));
                    frames_out.write(
                        reinterpret_cast<const char *>(frames[i].data_ptr<uint8_t>()),
                        frames[i].numel());
                }

                if (mask_acc[i][0] > 0.5f && disc_acc[i][0] > 0.5f)
                    pending[i].push_back({step, geometry[i], shells_ratio[i]});
            }

            const auto steps = env->step(
                wanted_frequency, TorchAction{continuous_action, discrete_action}.to_actions());

            const TorchStep torch_step(steps, vision_height, vision_width);
            state = torch_step.get_states();

            const auto step_rewards = torch_step.get_rewards().to(torch::kFloat).reshape({-1});
            const auto step_dones = torch_step.get_is_done().to(torch::kFloat).reshape({-1});
            const auto step_truncateds =
                torch_step.get_is_truncated().to(torch::kFloat).reshape({-1});
            for (int i = 0; i < nb_tanks; i++)
                rewards_csv << episode << ';' << step << ';' << i << ';'
                            << step_rewards[i].item<float>() << ';' << step_dones[i].item<float>()
                            << ';' << step_truncateds[i].item<float>() << '\n';

            const auto details = env->last_step_reward_details();
            for (int i = 0; i < nb_tanks; i++) {
                episode_kills += details[i].kill;
                episode_hits += details[i].nb_hits;
                episode_fired += details[i].nb_fired_shells;

                const int landed = details[i].nb_landed_shells;
                if (landed == 0) continue;

                const int hits = details[i].nb_hits;
                const float miss = details[i].sum_miss_distance / static_cast<float>(landed);

                for (int k = 0; k < landed && !pending[i].empty(); k++) {
                    const auto &shot = pending[i].front();
                    shots_csv << episode << ';' << i << ';' << shot.step << ';' << step << ';'
                              << shot.geometry.best_angle * to_degrees << ';'
                              << shot.geometry.best_distance << ';' << shot.geometry.distance << ';'
                              << shot.geometry.yaw * to_degrees << ';'
                              << shot.geometry.pitch * to_degrees << ';' << shot.shells << ';'
                              << landed << ';' << hits << ';' << miss << '\n';
                    pending[i].pop_front();
                }
            }
            is_done = env->is_episode_terminated();
            step++;
        }

        env->stop_drawing();
        std::cout << "episode " << episode << " : " << step << " steps, kill reward "
                  << episode_kills << ", hits " << episode_hits << ", fired " << episode_fired;
        if (network_target)
            std::cout << ", network designations " << network_designations
                      << ", same as nearest in view " << network_agreements;
        std::cout << std::endl;
    }

    return 0;
}
