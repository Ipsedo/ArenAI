//
// Created by samuel on 03/10/2025.
//

#include "./train.h"

#include <future>

#include <indicators/progress_bar.hpp>

#include <arenai_core/constants.h>

#include "./agents/sac.h"
#include "./core/train_environment.h"
#include "./core/train_gl_context.h"
#include "./utils/replay_buffer.h"
#include "./utils/saver.h"
#include "./utils/torch_converter.h"

bool is_episode_finish(const std::vector<bool> &already_done) {
    return std::accumulate(
               already_done.begin(), already_done.end(), 0,
               [](const int nb_done, const bool done) { return done ? nb_done + 1 : nb_done; })
           >= already_done.size() - 1;
}

std::string metrics_to_string(const std::vector<std::shared_ptr<Metric>> &metrics) {
    std::stringstream stream;

    stream << std::accumulate(
        metrics.begin(), metrics.end(), std::string(),
        [](std::string acc, const std::shared_ptr<Metric> &m) {
            return acc.append(", ").append(m->to_string());
        }) << " ";

    return stream.str();
}

void train_main(
    const float wanted_frequency, const ModelOptions &model_options,
    const TrainOptions &train_options) {
    std::cout << "Vision size : width=" << ENEMY_VISION_WIDTH << ", height=" << ENEMY_VISION_HEIGHT
              << std::endl;
    std::cout << "Proprioception size : " << ENEMY_PROPRIOCEPTION_SIZE << std::endl;
    std::cout << "Action size (continuous) : " << ENEMY_NB_CONTINUOUS_ACTION << std::endl;
    std::cout << "Action size (discrete) : " << ENEMY_NB_DISCRETE_ACTION << std::endl;

    torch::Device torch_device =
        train_options.cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    const auto env = std::make_unique<TrainTankEnvironment>(
        train_options.nb_tanks, train_options.android_asset_folder, wanted_frequency);

    auto agent = std::make_shared<SacAgent>(
        ENEMY_PROPRIOCEPTION_SIZE, ENEMY_NB_CONTINUOUS_ACTION, ENEMY_NB_DISCRETE_ACTION,
        train_options.learning_rate, model_options.hidden_size_sensors,
        model_options.hidden_size_actions, model_options.actor_hidden_size,
        model_options.critic_hidden_size, model_options.vision_channels,
        model_options.group_norm_nums, torch_device, train_options.metric_window_size,
        model_options.tau, model_options.gamma, model_options.initial_alpha);

    std::cout << "Parameters : " << agent->count_parameters() << std::endl;
    std::cout << "Target entropy (continuous) : " << agent->get_continuous_target_entropy()
              << std::endl;
    std::cout << "Target entropy (discrete) : " << agent->get_discrete_target_entropy()
              << std::endl;

    Saver saver(agent, train_options.output_folder, train_options.save_every);

    auto replay_buffer = std::make_unique<ReplayBuffer>(train_options.replay_buffer_size);

    Metric reward_metric("reward", train_options.metric_window_size, 6);
    Metric potential_metric("potential", train_options.metric_window_size, 2, true);

    auto sac_metrics = agent->get_metrics();

    std::cout << "Start training on " << train_options.nb_episodes << " episodes" << std::endl;

    int train_counter = 0;

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

    std::string sac_metric_p_bar_description = metrics_to_string(sac_metrics);

    auto gl_context = std::make_shared<TrainGlContext>();

    for (int episode_index = 0; episode_index < train_options.nb_episodes; episode_index++) {
        // set variable for episode
        bool is_done = false;
        std::vector already_done(train_options.nb_tanks, false);

        auto last_state = env->reset_physics();
        env->reset_drawables(gl_context);

        auto last_phi_vector = env->get_phi_vector();

        int episode_step_idx = 0;
        while (!is_done) {

            TorchAction torch_action;
            std::vector<Action> actions_for_env;

            const auto [vision, proprioception] = states_to_tensor(last_state);

            {
                torch::NoGradGuard no_grad_guard;
                agent->set_train(false);

                const auto
                    [continuous_action, continuous_log_proba, discrete_action, discrete_log_proba] =
                        agent->act(vision.to(torch_device), proprioception.to(torch_device));

                torch_action = {
                    continuous_action, continuous_log_proba, discrete_action, discrete_log_proba};

                actions_for_env = tensor_to_actions(continuous_action, discrete_action);
            }

            // step environment
            const auto steps = env->step(wanted_frequency, actions_for_env);
            const auto phi_vector = env->get_phi_vector();
            const auto is_truncated_vector = env->get_truncated_episodes();

            last_state.clear();
            last_state.reserve(train_options.nb_tanks);

            // save to replay buffer
            for (int i = 0; i < train_options.nb_tanks; i++) {
                const auto [next_state, reward, done] = steps[i];
                last_state.push_back(next_state);

                if (already_done[i]) continue;

                const bool is_done_and_not_truncated = done && !is_truncated_vector[i];

                const float potential_reward =
                    train_options.potential_reward_scale * (is_done_and_not_truncated ? 0.f : 1.f)
                    * (model_options.gamma * phi_vector[i] - last_phi_vector[i]);

                reward_metric.add(reward);
                potential_metric.add(potential_reward);

                const auto [next_vision, next_proprioception] = state_to_tensor(next_state);

                replay_buffer->add(
                    {{vision[i], proprioception[i]},
                     {torch_action.continuous_action[i], torch_action.continuous_log_proba[i],
                      torch_action.discrete_action[i], torch_action.discrete_log_proba[i]},
                     torch::tensor(
                         {reward + potential_reward}, torch::TensorOptions().dtype(torch::kFloat)),
                     torch::tensor(
                         {is_done_and_not_truncated}, torch::TensorOptions().dtype(torch::kBool)),
                     {next_vision, next_proprioception}});

                if (done && !already_done[i]) already_done[i] = true;
            }

            // check if it's time to train
            if (train_counter % train_options.train_every == train_options.train_every - 1
                && replay_buffer->size() >= train_options.batch_size * train_options.epochs) {
                agent->train(replay_buffer, train_options.epochs, train_options.batch_size);

                sac_metric_p_bar_description = metrics_to_string(sac_metrics);
            }

            is_done = is_episode_finish(already_done)
                      || episode_step_idx >= train_options.max_episode_steps;

            train_counter = (train_counter + 1) % train_options.train_every;
            episode_step_idx++;

            // attempt to save
            saver.attempt_save();

            // metric
            std::stringstream stream;
            stream << "Episode [" << episode_index << " / " << train_options.nb_episodes
                   << "] : " << reward_metric.to_string() << ", " << potential_metric.to_string()
                   << sac_metric_p_bar_description;

            p_bar.set_option(indicators::option::PrefixText{stream.str()});
            p_bar.print_progress();
        }

        last_state.clear();
        env->stop_drawing();

        p_bar.tick();
    }
}
