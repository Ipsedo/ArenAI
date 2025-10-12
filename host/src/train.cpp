//
// Created by samuel on 03/10/2025.
//

#include "./train.h"

#include <future>

#include <indicators/progress_bar.hpp>

#include "./core/train_environment.h"
#include "./core/train_gl_context.h"
#include "./networks/sac.h"
#include "./networks/target_update.h"
#include "./networks/truncated_normal.h"
#include "./utils/replay_buffer.h"
#include "./utils/saver.h"
#include "./utils/torch_converter.h"

bool is_all_done(const std::vector<std::tuple<State, Reward, IsFinish>> &result) {
    for (const auto &[s, r, is_done]: result)
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
        model_options.hidden_size_latent, model_options.hidden_size, torch_device,
        train_options.metric_window_size, model_options.tau, model_options.gamma);

    Saver saver(sac, train_options.output_folder, train_options.max_episode_steps);

    auto replay_buffer = std::make_shared<ReplayBuffer>(train_options.replay_buffer_size, 12345);

    Metric reward_metric("reward", train_options.metric_window_size);

    int counter = 0;

    indicators::ProgressBar p_bar{
        indicators::option::MinProgress{0},
        indicators::option::MaxProgress{
            train_options.nb_episodes * train_options.max_episode_steps},
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
        bool is_done = false;
        std::vector already_done(train_options.nb_tanks, false);
        auto state = env->reset_physics();
        env->reset_drawables(gl_context);

        int episode_step_idx = 0;
        while (!is_done) {
            // prepare state (remove dead agents)
            auto state_without_dead = state;

            for (int i = static_cast<int>(state_without_dead.size()) - 1; i >= 0; i--)
                if (already_done[i]) state_without_dead.erase(state.begin() + i);

            const auto [vision, proprioception] = state_core_to_tensor(state_without_dead);

            // sample action
            const auto [mu, sigma] =
                sac->act(vision.to(torch_device), proprioception.to(torch_device));
            const auto action = truncated_normal_sample(mu, sigma, -1.f, 1.f);
            const auto action_core = actions_tensor_to_core(action);

            auto action_promise = std::promise<std::vector<Action>>();
            action_promise.set_value(action_core);
            auto action_future = action_promise.get_future();

            // step environment
            const auto steps = env->step(1.f / 30.f, action_future);

            // prepare next state
            std::vector<State> next_state;
            next_state.reserve(steps.size());
            for (const auto &[s, r, d]: steps) next_state.push_back(s);
            const auto [next_vision, next_proprioception] = state_core_to_tensor(next_state);

            // save to replay buffer
            int index_action = 0;
            for (int i = 0; i < train_options.nb_tanks; i++) {
                if (already_done[i]) continue;

                const auto v = vision[i];
                const auto p = proprioception[i];

                const auto n_v = next_vision[i];
                const auto n_p = next_proprioception[i];

                const auto [_, r, d] = steps[i];

                replay_buffer->add(
                    {{v, p},
                     action[index_action],
                     torch::tensor(
                         r, torch::TensorOptions().device(torch_device).dtype(torch::kFloat)),
                     torch::tensor(
                         d, torch::TensorOptions().device(torch_device).dtype(torch::kBool)),
                     {n_v, n_p}});

                reward_metric.add(r);

                if (d) already_done[i] = true;

                index_action++;
            }

            // set new state
            state = next_state;

            // check if it's time to train
            if (counter % train_options.train_every == train_options.train_every - 1) {
                sac->train(replay_buffer, train_options.epochs, train_options.batch_size);

                auto metrics = sac->get_metrics();

                std::stringstream stream;
                stream << reward_metric.to_string()
                       << std::accumulate(
                              metrics.begin(), metrics.end(), std::string(),
                              [](std::string acc, const std::shared_ptr<Metric> &m) {
                                  return acc.append(", ").append(m->to_string());
                              })
                       << " ";
                p_bar.set_option(indicators::option::PrefixText{stream.str()});
                p_bar.tick();
            }

            is_done = is_all_done(steps) || episode_step_idx >= train_options.max_episode_steps;

            counter++;
            episode_step_idx++;

            // attempt to save
            saver.attempt_save();
        }
    }
}
