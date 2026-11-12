#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numbers>
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
#include "./networks/actor.h"
#include "./networks/q_function.h"
#include "./networks/recurrent/liquid_actor.h"
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

    // what the probe needs from an actor, whatever its family (ppo_liquid or sac)
    struct ProbeActorOutput {
        torch::Tensor mu, sigma, discrete;
        // empty for actors without an exposed feature layer (sac)
        torch::Tensor features, vision_features;
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

    const bool is_sac = !agent_config.contains("neuron_number");
    std::cout << "actor family : " << (is_sac ? "sac" : "ppo_liquid") << std::endl;

    if (is_sac) {
        const auto make_encoder = [&] {
            return std::make_shared<StateEncoder>(
                model::ENEMY_STATE_VECTOR_SIZE, agent_config.at("hidden_size_sensors").get<int>(),
                vision_channels, group_norm_nums);
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
                out.mu, out.sigma, out.discrete, actor_encoder->forward(vision, sensors),
                critic->encode(vision, sensors)};
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
            model::ENEMY_STATE_VECTOR_SIZE, model::ENEMY_NB_CONTINUOUS_ACTION,
            model::ENEMY_NB_DISCRETE_ACTION, agent_config.at("hidden_size_sensors").get<int>(),
            vision_channels, group_norm_nums, agent_config.at("neuron_number").get<int>(),
            agent_config.at("unfolding_steps").get<int>(), agent_config.at("delta_t").get<float>(),
            agent_config.at("initial_sigma").get<float>(), initial_discrete_probas);
        load_torch(save_folder, actor, "actor.pt");
        actor->to(device);
        actor->eval();

        const auto hidden_state = std::make_shared<LiquidHiddenState>(actor);
        reset_hidden = [hidden_state] { hidden_state->reset(); };
        act = [actor, hidden_state](const torch::Tensor &vision, const torch::Tensor &sensors) {
            const auto x_t = hidden_state->get(vision.size(0));
            auto out = actor->act(vision, sensors, x_t);
            hidden_state->set(out.next_x);
            return ProbeActorOutput{
                out.mu, out.sigma, out.discrete, out.features, out.vision_features};
        };
    }

    auto graphics_backend = view::make_vulkan_backend();
    std::cout << "Vulkan device : " << graphics_backend->renderer_info() << std::endl;

    const auto env = std::make_unique<TrainTankEnvironment>(
        std::move(graphics_backend), nb_tanks, resources_folder, wanted_frequency,
        max_episode_steps, vision_height, vision_width, parser.get<int>("--vision_num_threads"),
        action_repeat, agent_config.at("gamma").get<float>());

    std::ofstream csv(output);
    const auto &action_names = TorchAction::continuous_action_names();
    const std::vector aim_names(action_names.begin() + NB_DIRECTION_ACTIONS, action_names.end());

    csv << "episode;step;tank;distance;yaw;pitch;best_angle;best_distance;can_fire;shells;cooldown;"
           "p_fire;p_zoom;fire";
    for (const auto &name: action_names) csv << ";mu_" << name;
    for (const auto &prefix: {";sigma_", ";act_", ";dq_", ";dq1_", ";dq2_"})
        for (const auto &name: aim_names) csv << prefix << name;
    csv << ";turret_yaw\n";

    std::ofstream shots_csv(output.parent_path() / (output.stem().string() + "_shots.csv"));
    shots_csv
        << "episode;tank;fire_step;land_step;fire_best_angle;fire_best_distance;fire_distance;"
           "fire_yaw;fire_pitch;shells_at_fire;group_size;group_hits;group_miss\n";

    const auto features_path = parser.get<std::string>("--features");
    std::ofstream features_out;
    if (!features_path.empty()) features_out.open(features_path, std::ios::binary);

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
        std::vector<std::deque<PendingShot>> pending(nb_tanks);

        while (!is_done) {
            torch::Tensor continuous_action, discrete_action, mu, sigma, discrete, features,
                vision_features;
            {
                torch::NoGradGuard guard;

                auto out = act(state.get_vision().to(device), state.get_state().to(device));

                mu = out.mu.cpu();
                sigma = out.sigma.cpu();
                discrete = out.discrete.cpu();
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
            }

            const auto q_gradient = canon_q_gradient(state, mu, discrete);

            const auto geometry = env->aim_geometry();

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
                for (int a = 0; a < model::ENEMY_NB_CONTINUOUS_ACTION; a++)
                    csv << ';' << mu_acc[i][a];
                for (const auto &acc: {sigma_acc, cont_acc})
                    for (int a = NB_DIRECTION_ACTIONS; a < model::ENEMY_NB_CONTINUOUS_ACTION; a++)
                        csv << ';' << acc[i][a];
                for (int a = 0; a < 3 * static_cast<int>(aim_names.size()); a++)
                    csv << ';' << (q_gradient.defined() ? q_gradient[i][a].item<float>() : 0.f);
                csv << ';' << geometry[i].turret_yaw * to_degrees << '\n';

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

                if (mask_acc[i][0] > 0.5f && disc_acc[i][0] > 0.5f)
                    pending[i].push_back({step, geometry[i], shells_ratio[i]});
            }

            const auto steps = env->step(
                wanted_frequency, TorchAction{continuous_action, discrete_action}.to_actions());

            state = TorchStep(steps, vision_height, vision_width).get_states();

            const auto details = env->last_step_reward_details();
            for (int i = 0; i < nb_tanks; i++) {
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
        std::cout << "episode " << episode << " : " << step << " steps" << std::endl;
    }

    return 0;
}
