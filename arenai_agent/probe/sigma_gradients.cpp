#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <arenai_model/constants.h>
#include <arenai_model/shell_economy.h>
#include <arenai_view/backend.h>

#include "./agents/ppo_liquid/liquid_hidden_state.h"
#include "./agents/ppo_liquid/liquid_ppo_agent.h"
#include "./agents/ppo_liquid/liquid_ppo_collector.h"
#include "./agents/ppo_liquid/liquid_ppo_hyperparams.h"
#include "./agents/ppo_liquid/liquid_ppo_rollout_buffer.h"
#include "./agents/ppo_liquid/liquid_ppo_trainer.h"
#include "./agents/torch_types.h"
#include "./core/train_environment.h"
#include "./networks/recurrent/liquid_critic.h"
#include "./networks/vision.h"
#include "./networks_utils/torch_loader.h"

using namespace arenai;
using namespace arenai::agent;

namespace {
    struct StepRecord {
        std::vector<AimGeometry> geometries;
        std::vector<model::RewardDetail> details;
        std::vector<float> rewards;
        std::vector<bool> dones;
        std::vector<bool> truncateds;
    };

    LiquidPpoHyperParams read_params(const nlohmann::json &agent_config) {
        LiquidPpoHyperParams params{};
        for (const auto &field: liquid_ppo_cli_fields())
            std::visit(
                [&](const auto member) {
                    // a run older than a hyperparameter keeps its default
                    const auto name = field.name.substr(field.name.find_first_not_of('-'));
                    if (agent_config.contains(name))
                        params.*member =
                            agent_config.at(name).get<std::decay_t<decltype(params.*member)>>();
                },
                field.member);
        return params;
    }
}// namespace

int main(const int argc, char **argv) {
    argparse::ArgumentParser parser("arenai sigma gradients");

    parser.add_argument("--run_folder").required();
    parser.add_argument("--save").required();
    parser.add_argument("--resources_folder").required();
    parser.add_argument("--output").required();
    parser.add_argument("--nb_episodes").scan<'i', int>().default_value(2);
    parser.add_argument("--spawn_side").scan<'g', float>().default_value(1000.f);
    parser.add_argument("--critic_inputs").default_value(std::string());
    parser.add_argument("--frames").default_value(std::string());
    parser.add_argument("--frames_every").scan<'i', int>().default_value(3);
    parser.add_argument("--frames_fired").default_value(false).implicit_value(true);
    parser.add_argument("--kl_breakdown").default_value(std::string());
    parser.add_argument("--kl_record_every").scan<'i', int>().default_value(4);
    parser.add_argument("--cuda").default_value(false).implicit_value(true);
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

    const torch::Device device =
        parser.get<bool>("--cuda") ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    nlohmann::json config;
    std::ifstream(run_folder / "config.json") >> config;
    const auto &env_config = config.at("environment");
    const auto params = read_params(config.at("agent"));

    const int vision_height = env_config.at("vision_height").get<int>();
    const int vision_width = env_config.at("vision_width").get<int>();
    const float wanted_frequency = env_config.at("wanted_frequency").get<float>();
    const int nb_tanks = env_config.at("nb_tanks").get<int>();
    const model::ShellEconomy shell_economy{
        .initial_shells = env_config.at("shells").get<int>(),
        .shells_per_hit = env_config.at("shells_per_hit").get<int>(),
        .max_shells = env_config.at("max_shells").get<int>()};

    // the training triad of LiquidPpoTorchAgentFactory
    const auto actor = std::make_shared<LiquidActor>(
        vision_height, vision_width, model::ENEMY_STATE_VECTOR_SIZE,
        model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION,
        params.hidden_size_sensors, params.vision_channels, params.group_norm_nums,
        params.neuron_number, params.unfolding_steps, params.delta_t, params.initial_sigma,
        std::vector{params.initial_fire_proba, params.initial_zoom_proba});
    const auto hidden_state = std::make_shared<LiquidHiddenState>(actor);
    const auto rollout_buffer = std::make_shared<LiquidPpoRolloutBuffer>();
    const auto collector = std::make_shared<LiquidPpoStepCollector>(rollout_buffer, hidden_state);
    const auto agent =
        std::make_shared<TorchLiquidPpoAgent>(actor, hidden_state, device, collector);
    const auto trainer = std::make_shared<LiquidPpoTrainer>(
        actor, rollout_buffer, vision_height, vision_width, model::ENEMY_NB_CONTINUOUS_ACTION,
        model::ENEMY_NB_DISCRETE_ACTION, model::ENEMY_STATE_VECTOR_SIZE, params.actor_learning_rate,
        params.critic_learning_rate, params.hidden_size_sensors, params.vision_channels,
        params.group_norm_nums, params.neuron_number, params.unfolding_steps, params.delta_t,
        device, params.metric_window_size, params.gamma, params.gae_lambda, params.clip_epsilon,
        params.target_kl, params.grad_norm_max, params.initial_alpha, params.alpha_learning_rate,
        params.continuous_target_entropy, params.discrete_target_proba, params.actor_epochs,
        params.critic_epochs, params.rollout_size, params.actor_batch_size,
        params.actor_gpu_batch_size, params.critic_batch_size, params.critic_gpu_batch_size,
        params.chunk_size);
    trainer->load(save_folder);

    const auto critic_inputs_path = parser.get<std::string>("--critic_inputs");
    std::ofstream critic_inputs;
    std::shared_ptr<LiquidCritic> critic;
    if (!critic_inputs_path.empty()) {
        critic_inputs.open(critic_inputs_path, std::ios::binary);
        critic = std::make_shared<LiquidCritic>(
            vision_height, vision_width, model::ENEMY_STATE_VECTOR_SIZE, params.hidden_size_sensors,
            params.vision_channels, params.group_norm_nums, params.neuron_number,
            params.unfolding_steps, params.delta_t);
        load_torch(save_folder, critic, "critic.pt");
        critic->to(device);
        critic->eval();
    }

    const auto frames_path = parser.get<std::string>("--frames");
    const int frames_every = parser.get<int>("--frames_every");
    const bool frames_fired = parser.get<bool>("--frames_fired");
    std::ofstream frames_out;
    if (!frames_path.empty()) frames_out.open(frames_path, std::ios::binary);

    auto graphics_backend = view::make_vulkan_backend();
    std::cout << "Vulkan device : " << graphics_backend->renderer_info() << std::endl;

    const auto env = std::make_unique<TrainTankEnvironment>(
        std::move(graphics_backend), nb_tanks, resources_folder, wanted_frequency,
        config.at("train").at("max_episode_steps").get<int>(), vision_height, vision_width,
        parser.get<int>("--vision_num_threads"), env_config.at("action_repeat").get<int>());

    const auto names = TorchAction::continuous_action_names();

    std::ofstream csv(output);
    csv << "rollout;step;tank;armed;fire;best_angle;best_distance;distance;yaw;pitch;reward;"
           "hit;kill;received;death;done;truncated;advantage;nearest;position_x;position_z;"
           "nearest_x;nearest_z;canon_heading;chassis_heading;turret_yaw;fired_shells;landed_"
           "shells;"
           "nb_hits";
    for (const auto &prefix: {"mu_", "sigma_", "u_", "g_pg_", "g_h_"})
        for (const auto &name: names) csv << ';' << prefix << name;
    csv << '\n';

    const auto kl_breakdown_path = parser.get<std::string>("--kl_breakdown");
    const int kl_record_every = parser.get<int>("--kl_record_every");
    std::ofstream kl_csv;
    if (!kl_breakdown_path.empty()) {
        kl_csv.open(kl_breakdown_path);
        kl_csv << "rollout;step;sampled_kl";
        for (const auto &name: names) kl_csv << ";kl_" << name;
        for (const auto &name: names) kl_csv << ";kl_mean_" << name;
        for (const auto &name: names) kl_csv << ";shift_" << name;
        kl_csv << ";kl_fire;kl_zoom\n";
    }

    constexpr float to_degrees = 180.f / std::numbers::pi_v<float>;

    // one entry per step added to the rollout buffer, consumed with it
    std::deque<StepRecord> records;
    int rollout_index = 0;

    const auto write_rollout = [&] {
        const auto nb_steps = static_cast<int64_t>(rollout_buffer->nb_complete_steps());

        if (kl_csv.is_open()) {
            const auto breakdown = trainer->kl_breakdown(save_folder, kl_record_every);
            for (int64_t r = 0; breakdown.steps.defined() && r < breakdown.steps.size(0); r++) {
                kl_csv << rollout_index << ';' << breakdown.steps[r].item<float>() << ';'
                       << breakdown.sampled_kl[r].item<float>();
                for (const auto &kls:
                     {breakdown.continuous_kl, breakdown.continuous_mean_kl,
                      breakdown.continuous_mean_shift, breakdown.discrete_kl})
                    for (int64_t a = 0; a < kls.size(1); a++)
                        kl_csv << ';' << kls[r][a].item<float>();
                kl_csv << std::endl;
            }
            records.erase(records.begin(), records.begin() + nb_steps);
            std::cout << "rollout " << rollout_index << " : " << nb_steps << " steps" << std::endl;
            rollout_index++;
            return;
        }

        const auto gradients = trainer->sigma_gradients();

        if (gradients.steps.defined()) {
            const auto step_acc = gradients.steps.accessor<int64_t, 1>();
            const auto tank_acc = gradients.tanks.accessor<int64_t, 1>();
            const auto adv_acc = gradients.advantages.accessor<float, 1>();
            const auto armed_acc = gradients.fire_masks.accessor<float, 1>();
            const auto fire_acc = gradients.fires.accessor<float, 1>();
            const auto mu_acc = gradients.mu.accessor<float, 2>();
            const auto sigma_acc = gradients.sigma.accessor<float, 2>();
            const auto u_acc = gradients.pre_tanh.accessor<float, 2>();
            const auto pg_acc = gradients.surrogate.accessor<float, 2>();
            const auto h_acc = gradients.entropy.accessor<float, 2>();

            for (int64_t r = 0; r < gradients.steps.size(0); r++) {
                const auto &record = records[step_acc[r]];
                const auto tank = tank_acc[r];
                const auto &geometry = record.geometries[tank];
                const auto &detail = record.details[tank];
                csv << rollout_index << ';' << step_acc[r] << ';' << tank << ';' << armed_acc[r]
                    << ';' << fire_acc[r] << ';' << geometry.best_angle * to_degrees << ';'
                    << geometry.best_distance << ';' << geometry.distance << ';'
                    << geometry.yaw * to_degrees << ';' << geometry.pitch * to_degrees << ';'
                    << record.rewards[tank] << ';' << detail.hit << ';' << detail.kill << ';'
                    << detail.received << ';' << detail.death << ';' << record.dones[tank] << ';'
                    << record.truncateds[tank] << ';' << adv_acc[r] << ';' << geometry.nearest
                    << ';' << geometry.position_x << ';' << geometry.position_z << ';'
                    << geometry.nearest_x << ';' << geometry.nearest_z << ';'
                    << geometry.canon_heading * to_degrees << ';'
                    << geometry.chassis_heading * to_degrees << ';'
                    << geometry.turret_yaw * to_degrees << ';' << detail.nb_fired_shells << ';'
                    << detail.nb_landed_shells << ';' << detail.nb_hits;
                for (const auto &acc: {mu_acc, sigma_acc, u_acc, pg_acc, h_acc})
                    for (int64_t a = 0; a < static_cast<int64_t>(names.size()); a++)
                        csv << ';' << acc[r][a];
                csv << '\n';
            }
        }

        records.erase(records.begin(), records.begin() + nb_steps);
        std::cout << "rollout " << rollout_index << " : " << nb_steps << " steps" << std::endl;
        rollout_index++;
    };

    for (int episode = 0; episode < nb_episodes; episode++) {
        auto state = TorchState(
            env->reset(spawn_side, spawn_side, shell_economy), vision_height, vision_width);

        bool is_done = false;
        int episode_step = 0;
        while (!is_done) {
            StepRecord record{.geometries = env->aim_geometry()};

            const auto action = agent->act(state.to(device), true);
            torch::Tensor critic_features;
            if (critic) {
                torch::NoGradGuard guard;
                critic_features = critic->named_children()["vision_encoder"]
                                      ->as<ConvolutionNetwork>()
                                      ->forward(state.get_vision().to(device))
                                      .cpu()
                                      .contiguous();
            }
            const auto sensors = state.get_state().cpu().contiguous();
            const auto armed = state.get_fire_mask().cpu().contiguous();
            const auto fired = action.discrete_action.cpu().contiguous();

            // row : episode, step, tank, then the uint8 frame
            if (frames_out.is_open() && (frames_fired || episode_step % frames_every == 0)) {
                const auto frames = state.get_vision().cpu().contiguous();
                for (int i = 0; i < nb_tanks; i++) {
                    if (!record.geometries[i].alive) continue;
                    const bool shot =
                        armed[i][0].item<float>() > 0.5f && fired[i][0].item<float>() > 0.5f;
                    if (episode_step % frames_every != 0 && !shot) continue;
                    const float header[3] = {
                        static_cast<float>(episode), static_cast<float>(episode_step),
                        static_cast<float>(i)};
                    frames_out.write(reinterpret_cast<const char *>(header), sizeof(header));
                    frames_out.write(
                        reinterpret_cast<const char *>(frames[i].data_ptr<uint8_t>()),
                        frames[i].numel());
                }
            }
            const auto steps = env->step(wanted_frequency, action.to_actions());
            const TorchStep torch_step(steps, vision_height, vision_width);

            record.details = env->last_step_reward_details();
            for (const auto &[unused_state, reward, done, truncated]: steps) {
                record.rewards.push_back(reward);
                record.dones.push_back(done);
                record.truncateds.push_back(truncated);
            }

            // row : 16 floats, sensors, critic vision features
            for (int i = 0; critic && i < nb_tanks; i++) {
                const auto &geometry = record.geometries[i];
                const float header[16] = {
                    static_cast<float>(episode),
                    static_cast<float>(episode_step),
                    static_cast<float>(i),
                    geometry.alive ? 1.f : 0.f,
                    armed[i][0].item<float>(),
                    fired[i][0].item<float>(),
                    record.rewards[i],
                    record.dones[i] ? 1.f : 0.f,
                    record.truncateds[i] ? 1.f : 0.f,
                    geometry.distance,
                    geometry.yaw * to_degrees,
                    geometry.pitch * to_degrees,
                    geometry.best_angle * to_degrees,
                    geometry.best_distance,
                    record.details[i].hit,
                    record.details[i].kill};
                critic_inputs.write(reinterpret_cast<const char *>(header), sizeof(header));
                critic_inputs.write(
                    reinterpret_cast<const char *>(sensors[i].data_ptr<float>()),
                    sensors.size(1) * sizeof(float));
                critic_inputs.write(
                    reinterpret_cast<const char *>(critic_features[i].data_ptr<float>()),
                    critic_features.size(1) * sizeof(float));
            }
            episode_step++;
            records.push_back(std::move(record));

            collector->on_transition(
                torch_step.get_rewards(), env->last_step_aim_potentials(), torch_step.get_is_done(),
                torch_step.get_is_truncated());

            if (rollout_buffer->nb_complete_steps() >= static_cast<size_t>(params.rollout_size))
                write_rollout();

            is_done = env->is_episode_terminated();
            state = torch_step.get_states();
        }

        collector->on_episode_end(state);
        env->stop_drawing();
        std::cout << "episode " << episode << " done" << std::endl;
    }

    if (rollout_buffer->nb_complete_steps() >= static_cast<size_t>(params.chunk_size))
        write_rollout();

    return 0;
}
