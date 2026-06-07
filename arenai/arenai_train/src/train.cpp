//
// Created by samuel on 03/10/2025.
//

#include "./train.h"

#include <future>

#include <indicators/progress_bar.hpp>

#include <arenai_core/constants.h>
#include <arenai_train/replay_buffer.h>
#include <arenai_train/torch_converter.h>

#include "./agents/sac.h"
#include "./core/train_environment.h"
#include "./utils/saver.h"
#include "./view/train_gl_context.h"

void train_main(
    const float wanted_frequency, const EnvironmentOptions &environment_options,
    const ModelOptions &model_options, const TrainOptions &train_options) {

    auto gl_context = std::make_shared<TrainGlContext>();
    gl_context->make_current();// only for glGetString(GL_RENDERER)

    torch::Device torch_device =
        train_options.cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    std::cout << "OpenGL device : " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "PyTorch device : " << torch_device.str() << std::endl;

    std::cout << "Vision size : width=" << ENEMY_VISION_WIDTH << ", height=" << ENEMY_VISION_HEIGHT
              << std::endl;
    std::cout << "Proprioception size : " << ENEMY_PROPRIOCEPTION_SIZE << std::endl;
    std::cout << "Action size (continuous) : " << ENEMY_NB_CONTINUOUS_ACTION << std::endl;
    std::cout << "Action size (discrete) : " << ENEMY_NB_DISCRETE_ACTION << std::endl;

    const auto env = std::make_unique<TrainTankEnvironment>(
        gl_context, environment_options.nb_tanks, train_options.android_asset_folder,
        wanted_frequency, train_options.max_episode_steps);

    const float spawn_width_increase =
        (environment_options.final_spawn_width - environment_options.initial_spawn_width)
        / static_cast<float>(train_options.nb_episodes);
    const float spawn_height_increase =
        (environment_options.final_spawn_height - environment_options.initial_spawn_height)
        / static_cast<float>(train_options.nb_episodes);

    float spawn_width = environment_options.initial_spawn_width;
    float spawn_height = environment_options.initial_spawn_height;

    auto agent = std::make_shared<SacAgent>(
        ENEMY_PROPRIOCEPTION_SIZE, ENEMY_NB_CONTINUOUS_ACTION, ENEMY_NB_DISCRETE_ACTION,
        train_options.actor_learning_rate, train_options.critic_learning_rate,
        train_options.alpha_learning_rate, model_options.hidden_size_sensors,
        model_options.hidden_size_actions, model_options.actor_hidden_size,
        model_options.critic_hidden_size, model_options.vision_channels,
        model_options.group_norm_nums, torch_device, train_options.metric_window_size,
        model_options.tau, model_options.gamma, model_options.initial_alpha_continuous,
        model_options.initial_alpha_discrete);

    std::cout << "Parameters : " << agent->count_parameters() << std::endl;
    std::cout << "Target entropy (continuous) : " << agent->get_continuous_target_entropy()
              << std::endl;
    std::cout << "Target entropy (discrete) : " << agent->get_discrete_target_entropy()
              << std::endl;

    Saver saver(agent, train_options.output_folder, train_options.save_every);

    auto replay_buffer = std::make_unique<ReplayBuffer>(train_options.replay_buffer_size);

    // metrics
    auto reward_metric = std::make_shared<Metric>("reward", train_options.metric_window_size, 6);
    const auto sac_metrics = agent->get_metrics();
    const auto env_metrics = env->get_metrics();

    std::vector metrics = {reward_metric};
    metrics.insert(metrics.end(), env_metrics.begin(), env_metrics.end());
    metrics.insert(metrics.end(), sac_metrics.begin(), sac_metrics.end());

    // to detect when need train
    int train_counter = 0;

    // progress bar
    indicators::ProgressBar p_bar{
        indicators::option::MinProgress{0},
        indicators::option::MaxProgress{train_options.nb_episodes},
        indicators::option::BarWidth{10},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    for (int episode_index = 0; episode_index < train_options.nb_episodes; episode_index++) {
        const float spawn_side = std::sqrt(spawn_width * spawn_height);

        // set variable for episode
        bool is_done = false;

        auto last_states = env->reset_physics(spawn_width, spawn_height);
        env->reset_drawables(gl_context);

        while (!is_done) {

            TorchAction torch_action;
            std::vector<Action> actions_for_env;

            const auto [vision, proprioception] = states_to_tensor(last_states);

            {
                // action
                torch::NoGradGuard no_grad_guard;
                agent->set_train(false);

                const auto [continuous_action, discrete_action] =
                    agent->act(vision.to(torch_device), proprioception.to(torch_device));

                torch_action = {continuous_action, discrete_action};

                actions_for_env = tensor_to_actions(continuous_action, discrete_action);
            }

            // step environment
            const auto steps = env->step(wanted_frequency, actions_for_env);

            last_states.clear();
            last_states.reserve(environment_options.nb_tanks);

            // save to replay buffer
            for (int i = 0; i < environment_options.nb_tanks; i++) {
                const auto [next_state, reward, env_done] = steps[i];
                last_states.push_back(next_state);

                if (env->is_tank_factory_already_done(i)) continue;

                reward_metric->add(reward);

                const auto [next_vision, next_proprioception] = state_to_tensor(next_state);

                replay_buffer->add(
                    {{vision[i], proprioception[i]},
                     {torch_action.continuous_action[i], torch_action.discrete_action[i]},
                     torch::tensor({reward}, torch::TensorOptions().dtype(torch::kFloat)),
                     torch::tensor({env_done}, torch::TensorOptions().dtype(torch::kBool)),
                     {next_vision, next_proprioception}});
            }

            // check if it's time to train
            if (train_counter % train_options.train_every == train_options.train_every - 1
                && replay_buffer->size() >= train_options.batch_size * train_options.epochs)
                agent->train(replay_buffer, train_options.epochs, train_options.batch_size);

            // step ending stuff
            is_done = env->is_episode_terminated();

            train_counter = (train_counter + 1) % train_options.train_every;

            // attempt to save
            saver.attempt_save();

            // progress bar metrics display
            std::stringstream stream;
            stream << "\rEpisode [" << episode_index << " / " << train_options.nb_episodes
                   << "] (area = " << static_cast<int>(spawn_side)
                   << " m) : " << Metric::metrics_to_string(metrics);

            p_bar.set_option(indicators::option::PrefixText{stream.str()});
            p_bar.print_progress();
        }

        last_states.clear();
        env->stop_drawing();

        spawn_width += spawn_width_increase;
        spawn_height += spawn_height_increase;

        p_bar.tick();
    }
}
