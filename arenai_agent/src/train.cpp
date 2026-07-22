//
// Created by samuel on 03/10/2025.
//

#include "./train.h"

#include <future>

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>

#include <arenai_core/constants.h>

#include "./agents/sac/sac_factory.h"
#include "./core/train_environment.h"
#include "./metrics/mean_metric.h"
#include "./metrics/metric_saver.h"
#include "./networks_io/torch_saver.h"
#include "networks_utils/torch_converter.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    void train_main(
        const EnvironmentOptions &environment_options, const ModelOptions &model_options,
        const TrainOptions &train_options) {

        auto graphics_backend = view::make_vulkan_backend();

        torch::Device torch_device =
            train_options.cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

        std::cout << "Vulkan device : " << graphics_backend->renderer_info() << std::endl;
        std::cout << "PyTorch device : " << torch_device.str() << std::endl;

        std::cout << "Vision size : width=" << environment_options.vision_width
                  << ", height=" << environment_options.vision_height << std::endl;
        std::cout << "Proprioception size : " << model::ENEMY_PROPRIOCEPTION_SIZE << std::endl;
        std::cout << "Action size (continuous) : " << model::ENEMY_NB_CONTINUOUS_ACTION
                  << std::endl;
        std::cout << "Action size (discrete) : " << model::ENEMY_NB_DISCRETE_ACTION << std::endl;

        const auto env = std::make_unique<TrainTankEnvironment>(
            std::move(graphics_backend), environment_options.nb_tanks,
            train_options.resources_folder, environment_options.wanted_frequency,
            train_options.max_episode_steps, environment_options.vision_height,
            environment_options.vision_width, environment_options.num_threads);

        const float spawn_width_increase =
            (environment_options.final_spawn_width - environment_options.initial_spawn_width)
            / static_cast<float>(train_options.nb_episodes);
        const float spawn_height_increase =
            (environment_options.final_spawn_height - environment_options.initial_spawn_height)
            / static_cast<float>(train_options.nb_episodes);

        float spawn_width = environment_options.initial_spawn_width;
        float spawn_height = environment_options.initial_spawn_height;

        std::unique_ptr<AbstractTorchAgentFactory> agent_factory =
            std::make_unique<SacTorchAgentFactory>(
                environment_options.vision_height, environment_options.vision_width,
                model::ENEMY_PROPRIOCEPTION_SIZE, model::ENEMY_NB_CONTINUOUS_ACTION,
                model::ENEMY_NB_DISCRETE_ACTION, train_options.actor_learning_rate,
                train_options.critic_learning_rate, train_options.alpha_learning_rate,
                model_options.hidden_size_sensors, model_options.hidden_size_actions,
                model_options.actor_hidden_sizes, model_options.critic_hidden_sizes,
                model_options.vision_channels, model_options.group_norm_nums, torch_device,
                train_options.metric_window_size, model_options.tau, model_options.gamma,
                train_options.replay_buffer_size, train_options.train_every, train_options.epochs,
                train_options.batch_size);

        const auto agent = agent_factory->get_agent();
        const auto collector = agent_factory->get_collector();
        const auto trainer = agent_factory->get_trainer();

        std::cout << "Parameters : " << trainer->count_parameters() << std::endl;

        AgentSaver saver(trainer, train_options.output_folder, train_options.save_every);

        // metrics
        auto reward_mean_metric =
            std::make_shared<MeanMetric>("r", train_options.metric_window_size, 2, true);
        auto potential_mean_metric =
            std::make_shared<MeanMetric>("pr", train_options.metric_window_size, 2, true);

        const auto sac_metrics = trainer->get_metrics();
        const auto env_metrics = env->get_metrics();

        std::vector<std::shared_ptr<AbstractMetric>> metrics = {
            reward_mean_metric, potential_mean_metric};
        metrics.insert(metrics.end(), env_metrics.begin(), env_metrics.end());
        metrics.insert(metrics.end(), sac_metrics.begin(), sac_metrics.end());

        MetricCsvSaver metric_csv_saver(
            train_options.output_folder, metrics,
            static_cast<int>(30.f / environment_options.wanted_frequency));

        // progress bar
        indicators::ProgressBar p_bar{
            indicators::option::MinProgress{0},
            indicators::option::MaxProgress{train_options.nb_episodes},
            indicators::option::BarWidth{0},
            indicators::option::Start{"\r"},
            indicators::option::PrefixText{"\r"},
            indicators::option::Fill{""},
            indicators::option::Lead{""},
            indicators::option::Remainder{""},
            indicators::option::End{""},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true}};

        indicators::show_console_cursor(false);

        const int print_tqdm_bar_every =
            static_cast<int>(1.f / environment_options.wanted_frequency);
        int print_counter = 0;

        for (int episode_index = 0; episode_index < train_options.nb_episodes; episode_index++) {
            const float spawn_side = std::sqrt(spawn_width * spawn_height);

            // set variable for episode
            bool is_done = false;

            auto [vision, proprioception] = states_to_tensor(
                env->reset(spawn_width, spawn_height), environment_options.vision_height,
                environment_options.vision_width);
            auto last_phi_tensor = torch::tensor(env->get_phi_vector()).unsqueeze(1);

            while (!is_done) {

                std::vector<core::Action> actions_for_env;

                const auto [continuous_action, discrete_action] = agent->act(
                    {.vision = vision.to(torch_device),
                     .proprioception = proprioception.to(torch_device)});

                TorchAction torch_action = {
                    .continuous_action = continuous_action, .discrete_action = discrete_action};

                actions_for_env = tensor_to_actions(continuous_action, discrete_action);

                // step environment
                const auto steps = env->step(environment_options.wanted_frequency, actions_for_env);
                const auto phi_tensor = torch::tensor(env->get_phi_vector()).unsqueeze(1);

                const auto [torch_next_states, torch_rewards, torch_are_done, torch_are_truncated] =
                    steps_to_tensor(
                        steps, environment_options.vision_height, environment_options.vision_width);

                const auto is_not_terminal = torch::logical_not(
                    torch::logical_and(torch_are_done, torch::logical_not(torch_are_truncated)));
                const auto potential_rewards =
                    is_not_terminal * model_options.gamma * phi_tensor - last_phi_tensor;

                const auto torch_final_reward = torch_rewards + potential_rewards;

                // complete the pending transition - maybe train
                collector->on_transition(torch_final_reward, torch_are_done, torch_are_truncated);
                trainer->step();

                // step ending stuff
                is_done = env->is_episode_terminated();
                last_phi_tensor = phi_tensor;

                vision = torch_next_states.vision;
                proprioception = torch_next_states.proprioception;

                print_counter = (print_counter + 1) % print_tqdm_bar_every;

                // attempt to save
                saver.attempt_save();
                metric_csv_saver.attempt_append_to_csv();

                // metrics
                reward_mean_metric->add(torch_rewards.mean().item<float>());
                potential_mean_metric->add(potential_rewards.mean().item<float>());

                // progress bar metrics display
                if (print_counter == print_tqdm_bar_every - 1) {
                    std::stringstream stream;
                    stream << "- " << episode_index << " (" << static_cast<int>(spawn_side)
                           << "m) : " << AbstractMetric::metrics_to_string(metrics);

                    p_bar.set_option(indicators::option::PostfixText{stream.str()});
                    p_bar.print_progress();
                }
            }

            // close episode
            collector->on_episode_end({.vision = vision, .proprioception = proprioception});

            env->stop_drawing();

            spawn_width += spawn_width_increase;
            spawn_height += spawn_height_increase;

            p_bar.tick();
        }
    }

}// namespace arenai::agent
