//
// Created by samuel on 03/10/2025.
//

#include "./train.h"

#include <future>

#include <indicators/progress_bar.hpp>

#include <arenai_core/constants.h>

#include "./core/train_environment.h"
#include "./core/train_gl_context.h"
#include "./networks/sac.h"
#include "./networks/target_update.h"
#include "./networks/truncated_normal.h"
#include "./utils/replay_buffer.h"
#include "./utils/saver.h"
#include "./utils/torch_converter.h"

bool is_all_done(const std::vector<bool> &already_done) {
    for (const auto &is_done: already_done)
        if (!is_done) return false;
    return true;
}

void train_main(const ModelOptions &model_options, const TrainOptions &train_options) {
    torch::Device torch_device =
        train_options.cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    const auto env = std::make_unique<TrainTankEnvironment>(
        train_options.nb_tanks, train_options.android_asset_folder);
    auto sac = std::make_shared<SacNetworks>(
        ENEMY_PROPRIOCEPTION_SIZE, ENEMY_NB_ACTION, train_options.learning_rate,
        model_options.hidden_size_sensors, model_options.hidden_size_actions,
        model_options.hidden_size, torch_device, train_options.metric_window_size,
        model_options.tau, model_options.gamma, model_options.initial_alpha);

    Saver saver(sac, train_options.output_folder, train_options.save_every);

    auto replay_buffer = std::make_unique<ReplayBuffer>(train_options.replay_buffer_size, 12345);

    Metric reward_metric("reward", train_options.metric_window_size);
    Metric potential_reward_metric("potential_reward", train_options.metric_window_size);

    int counter = 0;

    indicators::ProgressBar p_bar{
        indicators::option::MinProgress{0},
        indicators::option::MaxProgress{train_options.nb_episodes},
        indicators::option::BarWidth{30},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    auto gl_context = std::make_shared<TrainGlContext>();

    for (int episode_index = 0; episode_index < train_options.nb_episodes; episode_index++) {
        // set variable for episode
        bool is_done = false;
        std::vector already_done(train_options.nb_tanks, false);

        auto last_state = env->reset_physics();
        env->reset_drawables(gl_context);

        int episode_step_idx = 0;
        while (!is_done) {

            torch::Tensor actions;

            const auto [vision, proprioception] = states_to_tensor(last_state);

            auto actions_future = std::async([&]() {
                torch::NoGradGuard no_grad_guard;

                sac->train(false);

                const auto [mu, sigma] =
                    sac->act(vision.to(torch_device), proprioception.to(torch_device));
                actions = truncated_normal_sample(mu, sigma, -1.f, 1.f).cpu();

                return tensor_to_actions(actions);
            });

            const auto potential_rewards = env->get_potential_rewards();

            // step environment
            const auto steps = env->step(1.f / 30.f, actions_future);

            const auto next_potential_rewards = env->get_potential_rewards();

            last_state.clear();
            last_state.reserve(train_options.nb_tanks);

            // save to replay buffer
            for (int i = 0; i < train_options.nb_tanks; i++) {
                const auto [next_state, reward, done] = steps[i];
                last_state.push_back(next_state);

                if (already_done[i]) continue;

                const auto [next_vision, next_proprioception] = state_to_tensor(next_state);

                reward_metric.add(reward);
                potential_reward_metric.add(potential_rewards[i]);

                replay_buffer->add(
                    {{vision[i], proprioception[i]},
                     actions[i],
                     torch::tensor(
                         reward + (done ? 0.f : model_options.gamma * next_potential_rewards[i])
                             - potential_rewards[i],
                         torch::TensorOptions().dtype(torch::kFloat))
                         .unsqueeze(0),
                     torch::tensor(done, torch::TensorOptions().dtype(torch::kBool)).unsqueeze(0),
                     {next_vision, next_proprioception}});

                if (done && !already_done[i]) already_done[i] = true;
            }

            // check if it's time to train
            if (counter % train_options.train_every == train_options.train_every - 1)
                sac->train(replay_buffer, train_options.epochs, train_options.batch_size);

            // progress bar
            auto metrics = sac->get_metrics();

            std::stringstream stream;
            stream << reward_metric.to_string() << ", " << potential_reward_metric.to_string()
                   << std::accumulate(
                          metrics.begin(), metrics.end(), std::string(),
                          [](std::string acc, const std::shared_ptr<Metric> &m) {
                              return acc.append(", ").append(m->to_string());
                          })
                   << " ";
            p_bar.set_option(indicators::option::PrefixText{stream.str()});
            p_bar.print_progress();

            is_done =
                is_all_done(already_done) || episode_step_idx >= train_options.max_episode_steps;

            counter++;
            episode_step_idx++;

            // attempt to save
            saver.attempt_save();
        }

        last_state.clear();
        env->stop_drawing();
        p_bar.tick();
    }
}
