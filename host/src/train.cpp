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

bool is_all_done(const std::vector<std::tuple<State, Reward, IsFinish>> &result) {
    for (const auto &[s, r, is_done]: result)
        if (!is_done) return false;
    return true;
}

void train_main(
    const std::filesystem::path &output_folder, const std::filesystem::path &android_assets_path) {
    constexpr float learning_rate = 1e-4f;
    int batch_size = 32;
    int nb_episodes = 2048;
    int train_every = 32;
    constexpr int max_episode_steps = 30 * 60 * 5;
    constexpr int replay_buffer_size = max_episode_steps * 10;
    int nb_tanks = 8;

    bool cuda = true;

    torch::Device torch_device = cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    const auto env = std::make_unique<TrainTankEnvironment>(nb_tanks, android_assets_path);
    auto sac = SacNetworks(
        ENEMY_PROPRIOCEPTION_SIZE, ENEMY_NB_ACTION, learning_rate, 160, 320, torch_device, 64);

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
        const auto state = env->reset_physics();
        env->reset_drawables(std::make_shared<TrainGlContext>());

        while (!is_done) {
            const auto [vision, proprioception] = state_core_to_tensor(state);

            const auto [mu, sigma] =
                sac.act(vision.to(torch_device), proprioception.to(torch_device));
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

            for (int i = 0; i < state.size(); i++) {
                const auto v = vision[i];
                const auto p = proprioception[i];

                const auto n_v = next_vision[i];
                const auto n_p = next_proprioception[i];

                const auto [_, r, d] = steps[i];

                TorchStep torch_step{
                    {v, p},
                    action[i],
                    torch::tensor(
                        r, torch::TensorOptions().device(torch_device).dtype(torch::kFloat)),
                    torch::tensor(
                        d, torch::TensorOptions().device(torch_device).dtype(torch::kBool)),
                    {n_v, n_p}};

                replay_buffer->add(torch_step);

                reward_metric.add(r);
            }

            if (counter % train_every == train_every - 1) {
                sac.train(replay_buffer, batch_size);
                auto metrics = sac.get_metrics();
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

            is_done = is_all_done(steps);
            counter++;
        }
    }
}
