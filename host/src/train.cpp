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
#include "./utils/linux_file_reader.h"
#include "./utils/replay_buffer.h"
#include "./utils/torch_converter.h"
#include "utils/saver.h"

bool is_all_done(const std::vector<std::tuple<State, Reward, IsFinish>> &result) {
    for (const auto &[s, r, is_done]: result)
        if (!is_done) return false;
    return true;
}

void train_main(
    const std::filesystem::path &output_folder, const std::filesystem::path &android_assets_path) {
    constexpr float learning_rate = 1e-3f;
    int batch_size = 256;
    int nb_episodes = 2048;
    int train_every = 32;
    constexpr int max_episode_steps = 30 * 60 * 5;
    constexpr int replay_buffer_size = 16384;
    int nb_tanks = 8;

    bool cuda = true;

    torch::Device torch_device = cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    const auto env = std::make_unique<TrainTankEnvironment>(nb_tanks, android_assets_path);
    auto sac = std::make_shared<SacNetworks>(
        ENEMY_PROPRIOCEPTION_SIZE, ENEMY_NB_ACTION, learning_rate, 160, 320, torch_device, 64,
        1e-3f, 0.95f);

    Saver saver(
        sac, "/home/samuel/StudioProjects/PhyVR/host/outputs/train_sac_first_try",
        max_episode_steps);

    auto replay_buffer = std::make_shared<ReplayBuffer>(replay_buffer_size, 12345);

    Metric reward_metric("reward", 64);

    int counter = 0;

    indicators::ProgressBar p_bar{
        indicators::option::MinProgress{0},
        indicators::option::MaxProgress{nb_episodes * max_episode_steps},
        indicators::option::BarWidth{30},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}};

    for (int episode_index = 0; episode_index < nb_episodes; episode_index++) {
        bool is_done = false;
        std::vector already_done(nb_tanks, false);
        auto state = env->reset_physics();
        auto state_without_dead = state;
        env->reset_drawables(std::make_shared<TrainGlContext>());

        int episode_step_idx = 0;
        while (!is_done) {
            const auto [vision, proprioception] = state_core_to_tensor(state_without_dead);

            const auto [mu, sigma] =
                sac->act(vision.to(torch_device), proprioception.to(torch_device));
            const auto action = truncated_normal_sample(mu, sigma, -1.f, 1.f);
            const auto action_core = actions_tensor_to_core(action);

            auto action_promise = std::promise<std::vector<Action>>();
            action_promise.set_value(action_core);
            auto action_future = action_promise.get_future();

            const auto steps = env->step(1.f / 30.f, action_future);

            std::vector<State> next_state;
            next_state.reserve(steps.size());

            for (const auto &[s, r, d]: steps) next_state.push_back(s);

            const auto [next_vision, next_proprioception] = state_core_to_tensor(next_state);

            int index = 0;
            for (int i = 0; i < nb_tanks; i++) {
                if (already_done[i]) continue;

                const auto v = vision[index];
                const auto p = proprioception[index];

                const auto n_v = next_vision[index];
                const auto n_p = next_proprioception[index];

                const auto [_, r, d] = steps[index];

                TorchStep torch_step{
                    {v, p},
                    action[index],
                    torch::tensor(
                        r, torch::TensorOptions().device(torch_device).dtype(torch::kFloat)),
                    torch::tensor(
                        d, torch::TensorOptions().device(torch_device).dtype(torch::kBool)),
                    {n_v, n_p}};

                replay_buffer->add(torch_step);

                reward_metric.add(r);

                if (d) already_done[i] = true;

                index++;
            }

            state_without_dead = state = next_state;

            for (int i = state_without_dead.size() - 1; i >= 0; i--) {
                const auto [s, r, d] = steps[i];
                if (d) state_without_dead.erase(state.begin() + i);
            }

            if (counter % train_every == train_every - 1) {
                sac->train(replay_buffer, batch_size);

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

            is_done = is_all_done(steps) || episode_step_idx >= max_episode_steps;

            counter++;
            episode_step_idx++;

            saver.attempt_save();
        }
    }
}
