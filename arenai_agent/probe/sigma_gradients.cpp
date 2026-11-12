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
    parser.add_argument("--cuda").default_value(false).implicit_value(true);
    parser.add_argument("--kl_by_group").default_value(false).implicit_value(true);
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
    const bool kl_by_group = parser.get<bool>("--kl_by_group");

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
        model::ENEMY_STATE_VECTOR_SIZE, model::ENEMY_NB_CONTINUOUS_ACTION,
        model::ENEMY_NB_DISCRETE_ACTION, params.hidden_size_sensors, params.vision_channels,
        params.group_norm_nums, params.neuron_number, params.unfolding_steps, params.delta_t,
        params.initial_sigma, std::vector{params.initial_fire_proba, params.initial_zoom_proba});
    const auto hidden_state = std::make_shared<LiquidHiddenState>(actor);
    const auto rollout_buffer = std::make_shared<LiquidPpoRolloutBuffer>();
    const auto collector = std::make_shared<LiquidPpoStepCollector>(rollout_buffer, hidden_state);
    const auto agent =
        std::make_shared<TorchLiquidPpoAgent>(actor, hidden_state, device, collector);
    const auto trainer = std::make_shared<LiquidPpoTrainer>(
        actor, rollout_buffer, model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION,
        model::ENEMY_STATE_VECTOR_SIZE, params.actor_learning_rate, params.critic_learning_rate,
        params.hidden_size_sensors, params.vision_channels, params.group_norm_nums,
        params.neuron_number, params.unfolding_steps, params.delta_t, device,
        params.metric_window_size, params.gamma, params.gae_lambda, params.aim_gae_lambda,
        params.clip_epsilon, params.target_kl, params.grad_norm_max, params.initial_alpha,
        params.alpha_learning_rate, params.continuous_target_entropy, params.discrete_target_proba,
        params.actor_epochs, params.critic_epochs, params.rollout_size, params.actor_batch_size,
        params.actor_gpu_batch_size, params.critic_batch_size, params.critic_gpu_batch_size,
        params.chunk_size);
    const auto load_save = [&] {
        trainer->load(save_folder);
        if (kl_by_group) trainer->load_actor_optimizer(save_folder);
    };
    load_save();

    auto graphics_backend = view::make_vulkan_backend();
    std::cout << "Vulkan device : " << graphics_backend->renderer_info() << std::endl;

    const auto env = std::make_unique<TrainTankEnvironment>(
        std::move(graphics_backend), nb_tanks, resources_folder, wanted_frequency,
        config.at("train").at("max_episode_steps").get<int>(), vision_height, vision_width,
        parser.get<int>("--vision_num_threads"), env_config.at("action_repeat").get<int>(),
        params.gamma);

    const auto names = TorchAction::continuous_action_names();

    std::ofstream csv(output);
    if (kl_by_group) csv << "rollout;epoch;step;joint;move;canon;discrete\n";
    else {
        csv << "rollout;step;tank;armed;fire;best_angle;best_distance;distance;yaw;pitch;reward;"
               "hit;kill;received;death;done;truncated;advantage;aim_advantage";
        for (const auto &prefix: {"mu_", "sigma_", "u_", "g_pg_", "g_h_"})
            for (const auto &name: names) csv << ';' << prefix << name;
        csv << '\n';
    }

    constexpr float to_degrees = 180.f / std::numbers::pi_v<float>;

    // one entry per step added to the rollout buffer, consumed with it
    std::deque<StepRecord> records;
    int rollout_index = 0;

    const auto write_kl_by_group = [&] {
        const auto nb_steps = static_cast<int64_t>(rollout_buffer->nb_complete_steps());

        for (const auto &[epoch, step, joint, move, canon, discrete]: trainer->kl_by_group())
            csv << rollout_index << ';' << epoch << ';' << step << ';' << joint << ';' << move
                << ';' << canon << ';' << discrete << '\n';

        load_save();

        records.erase(records.begin(), records.begin() + nb_steps);
        std::cout << "rollout " << rollout_index << " : " << nb_steps << " steps" << std::endl;
        rollout_index++;
    };

    const auto write_rollout = [&] {
        if (kl_by_group) return write_kl_by_group();

        const auto nb_steps = static_cast<int64_t>(rollout_buffer->nb_complete_steps());
        const auto gradients = trainer->sigma_gradients();

        if (gradients.steps.defined()) {
            const auto step_acc = gradients.steps.accessor<int64_t, 1>();
            const auto tank_acc = gradients.tanks.accessor<int64_t, 1>();
            const auto adv_acc = gradients.advantages.accessor<float, 1>();
            const auto aim_adv_acc = gradients.aim_advantages.accessor<float, 1>();
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
                    << record.truncateds[tank] << ';' << adv_acc[r] << ';' << aim_adv_acc[r];
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
        while (!is_done) {
            StepRecord record{.geometries = env->aim_geometry()};

            const auto steps =
                env->step(wanted_frequency, agent->act(state.to(device), true).to_actions());
            const TorchStep torch_step(steps, vision_height, vision_width);

            record.details = env->last_step_reward_details();
            for (const auto &[unused_state, reward, done, truncated]: steps) {
                record.rewards.push_back(reward);
                record.dones.push_back(done);
                record.truncateds.push_back(truncated);
            }
            records.push_back(std::move(record));

            collector->on_transition(
                torch_step.get_rewards(), env->last_step_aim_rewards(), torch_step.get_is_done(),
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
