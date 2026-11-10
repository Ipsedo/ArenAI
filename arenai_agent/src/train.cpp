#include "./train.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <future>
#include <string>

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <nlohmann/json.hpp>

#include <arenai_model/constants.h>

#include "./agents/torch_types.h"
#include "./core/curriculum.h"
#include "./core/train_environment.h"
#include "./metrics/last_metric.h"
#include "./metrics/metric_saver.h"
#include "./networks_utils/torch_saver.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    namespace {
        void save_run_config(
            const EnvironmentOptions &environment_options, const TrainOptions &train_options,
            const nlohmann::json &agent_config) {

            const nlohmann::json config = {
                {"train",
                 {{"output_folder", train_options.output_folder.string()},
                  {"resources_folder", train_options.resources_folder.string()},
                  {"max_episode_steps", train_options.max_episode_steps},
                  {"nb_episodes", train_options.nb_episodes},
                  {"save_every", train_options.save_every},
                  {"cuda", train_options.cuda}}},
                {"environment",
                 {{"wanted_frequency", environment_options.wanted_frequency},
                  {"nb_tanks", environment_options.nb_tanks},
                  {"vision_height", environment_options.vision_height},
                  {"vision_width", environment_options.vision_width},
                  {"initial_spawn_width", environment_options.initial_spawn_width},
                  {"initial_spawn_height", environment_options.initial_spawn_height},
                  {"final_spawn_width", environment_options.final_spawn_width},
                  {"final_spawn_height", environment_options.final_spawn_height},
                  {"shells", environment_options.shell_economy.initial_shells},
                  {"shells_per_hit", environment_options.shell_economy.shells_per_hit},
                  {"max_shells", environment_options.shell_economy.max_shells},
                  {"curriculum_delta", environment_options.curriculum_delta},
                  {"curriculum_ratio_low", environment_options.curriculum_ratio_low},
                  {"curriculum_ratio_high", environment_options.curriculum_ratio_high},
                  {"curriculum_probe_window", environment_options.curriculum_probe_window},
                  {"curriculum_boundary_proba", environment_options.curriculum_boundary_proba},
                  {"vision_num_threads", environment_options.num_threads},
                  {"action_repeat", environment_options.action_repeat}}},
                {"agent", agent_config}};

            std::filesystem::create_directories(train_options.output_folder);

            std::ofstream stream(train_options.output_folder / "config.json");
            stream << config.dump(4) << std::endl;
        }
    }// namespace

    void train_main(
        const EnvironmentOptions &environment_options, const TrainOptions &train_options,
        const std::unique_ptr<AbstractTorchAgentFactory> &agent_factory) {

        auto graphics_backend = view::make_vulkan_backend();

        torch::Device torch_device =
            train_options.cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

        std::cout << "Vulkan device : " << graphics_backend->renderer_info() << std::endl;
        std::cout << "PyTorch device : " << torch_device.str() << std::endl;

        std::cout << "Vision size : width=" << environment_options.vision_width
                  << ", height=" << environment_options.vision_height << std::endl;
        std::cout << "State vector size : " << model::ENEMY_STATE_VECTOR_SIZE << std::endl;
        std::cout << "Action size (continuous) : " << model::ENEMY_NB_CONTINUOUS_ACTION
                  << std::endl;
        std::cout << "Action size (discrete) : " << model::ENEMY_NB_DISCRETE_ACTION << std::endl;

        const auto env = std::make_unique<TrainTankEnvironment>(
            std::move(graphics_backend), environment_options.nb_tanks,
            train_options.resources_folder, environment_options.wanted_frequency,
            train_options.max_episode_steps, environment_options.vision_height,
            environment_options.vision_width, environment_options.num_threads,
            environment_options.action_repeat,
            agent_factory->get_config().at("gamma").get<float>());

        const float initial_spawn_side = std::sqrt(
            environment_options.initial_spawn_width * environment_options.initial_spawn_height);
        const float final_spawn_side = std::sqrt(
            environment_options.final_spawn_width * environment_options.final_spawn_height);

        const float curriculum_delta_progress =
            final_spawn_side > initial_spawn_side
                ? environment_options.curriculum_delta / (final_spawn_side - initial_spawn_side)
                : 0.f;

        constexpr std::uint64_t curriculum_seed = 1337;
        ProgressCurriculum curriculum(
            curriculum_delta_progress, environment_options.curriculum_ratio_low,
            environment_options.curriculum_ratio_high, environment_options.curriculum_probe_window,
            environment_options.curriculum_boundary_proba, curriculum_seed);

        const auto agent = agent_factory->get_agent();
        const auto collector = agent_factory->get_collector();
        const auto trainer = agent_factory->get_trainer();

        std::cout << "Parameters : " << trainer->count_parameters() << std::endl;

        save_run_config(environment_options, train_options, agent_factory->get_config());

        AgentSaver saver(trainer, train_options.output_folder, train_options.save_every);

        // metrics

        const auto trainer_metrics = trainer->get_metrics();
        const auto env_metrics = env->get_metrics();

        // curriculum observability: the side played this episode and the current bound
        const auto spawn_side_metric = std::make_shared<LastMetric>("side", 0);
        const auto spawn_bound_metric = std::make_shared<LastMetric>("D", 0);

        std::vector<std::shared_ptr<AbstractMetric>> metrics;
        metrics.insert(metrics.end(), env_metrics.begin(), env_metrics.end());
        metrics.push_back(spawn_side_metric);
        metrics.push_back(spawn_bound_metric);
        metrics.insert(metrics.end(), trainer_metrics.begin(), trainer_metrics.end());

        const auto env_csv_only_metrics = env->get_csv_only_metrics();
        const auto trainer_csv_only_metrics = trainer->get_csv_only_metrics();
        auto csv_metrics = metrics;
        csv_metrics.insert(
            csv_metrics.end(), env_csv_only_metrics.begin(), env_csv_only_metrics.end());
        csv_metrics.insert(
            csv_metrics.end(), trainer_csv_only_metrics.begin(), trainer_csv_only_metrics.end());

        MetricCsvSaver metric_csv_saver(
            train_options.output_folder, csv_metrics,
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
            const float progress = curriculum.sample_progress();

            const float spawn_width = std::lerp(
                environment_options.initial_spawn_width, environment_options.final_spawn_width,
                progress);
            const float spawn_height = std::lerp(
                environment_options.initial_spawn_height, environment_options.final_spawn_height,
                progress);
            const float spawn_side = std::sqrt(spawn_width * spawn_height);

            spawn_side_metric->add(spawn_side);
            spawn_bound_metric->add(
                std::lerp(initial_spawn_side, final_spawn_side, curriculum.upper_bound()));

            // set variable for episode
            bool is_done = false;

            auto state = TorchState(
                env->reset(spawn_width, spawn_height, environment_options.shell_economy),
                environment_options.vision_height, environment_options.vision_width);

            while (!is_done) {

                std::vector<core::Action> actions_for_env;

                actions_for_env = agent->act(state.to(torch_device), true).to_actions();

                // step environment
                const auto steps = env->step(environment_options.wanted_frequency, actions_for_env);

                const TorchStep torch_step(
                    steps, environment_options.vision_height, environment_options.vision_width);

                // complete the pending transition - maybe train
                collector->on_transition(
                    torch_step.get_rewards(), torch_step.get_is_done(),
                    torch_step.get_is_truncated());
                trainer->step();

                // step ending stuff
                is_done = env->is_episode_terminated();

                state = torch_step.get_states();

                print_counter = (print_counter + 1) % print_tqdm_bar_every;

                // attempt to save
                saver.attempt_save();
                metric_csv_saver.attempt_append_to_csv();

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
            collector->on_episode_end(state);

            env->stop_drawing();

            curriculum.on_episode_end(env->episode_nb_fires(), env->episode_nb_hits());

            p_bar.tick();
        }
    }

}// namespace arenai::agent
