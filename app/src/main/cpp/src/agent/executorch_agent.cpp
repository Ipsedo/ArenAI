//
// Created by samuel on 03/10/2025.
//

#include "./executorch_agent.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>

#include "./loader.h"

#define SIGMA_MIN 1e-6f
#define SIGMA_MAX 1e6f
#define ALPHA_BETA_BOUND 5.f

/*
 * ExecuTorch
 */

ExecuTorchAgent::ExecuTorchAgent(android_app *app, const std::string &pte_asset_path)
    : actor_module(copy_asset_to_files(
        app->activity->assetManager, pte_asset_path, app->activity->internalDataPath)),
      dev(), rng(dev()) {}

std::vector<Action> ExecuTorchAgent::act(const std::vector<State> &state) {
    const auto N = static_cast<int64_t>(state.size());
    const int64_t C = 3;
    const int64_t H = ENEMY_VISION_SIZE;
    const int64_t W = ENEMY_VISION_SIZE;

    auto buffer_vision = std::vector<float>(static_cast<size_t>(N * C * H * W));
    auto buffer_proprioception = std::vector<float>(static_cast<size_t>(N * ENEMY_PROPRIOCEPTION_SIZE));

    unsigned long idx_vision = 0;
    size_t idx_proprioception = 0;
    for (int64_t n = 0; n < N; n++) {
        const auto &[img, proprioception] = state[static_cast<size_t>(n)];
        for (int64_t c = 0; c < C; c++) {
            const auto &plane = img[static_cast<size_t>(c)];
            for (int64_t h = 0; h < H; h++) {
                const auto &row = plane[static_cast<size_t>(h)];
                for (int64_t w = 0; w < W; w++) {
                    buffer_vision[idx_vision] = 2.f * static_cast<float>(row[w]) / 255.f - 1.f;
                    idx_vision += 1;
                }
            }
        }
        std::memcpy(
            buffer_proprioception.data() + idx_proprioception, proprioception.data(),
            sizeof(float) * static_cast<size_t>(ENEMY_PROPRIOCEPTION_SIZE));
        idx_proprioception += static_cast<size_t>(ENEMY_PROPRIOCEPTION_SIZE);
    }

    /*std::memcpy(
        dst_p + batch_idx * ENEMY_PROPRIOCEPTION_SIZE, s.proprioception.data(),
        sizeof(float) * static_cast<size_t>(ENEMY_PROPRIOCEPTION_SIZE));*/

    const auto dtype = torch::executor::ScalarType::Float;

    auto vision_tensor = executorch::extension::from_blob(
        static_cast<void *>(buffer_vision.data()), {static_cast<int>(N), C, H, W}, dtype);
    auto proprioception_tensor = executorch::extension::from_blob(
        static_cast<void *>(buffer_proprioception.data()),
        {static_cast<int>(N), ENEMY_PROPRIOCEPTION_SIZE}, dtype);

    auto output = actor_module.forward({vision_tensor, proprioception_tensor});

    if (!output.ok()) throw std::runtime_error(executorch::runtime::to_string(output.error()));

    auto sampled_actions = output->at(0).toTensor().const_data_ptr<float>();

    std::vector<Action> actions(N);
    for (int i = 0; i < N; i++) {
        auto start_idx = i * ENEMY_NB_ACTION;
        joystick joystick_direction{sampled_actions[start_idx], sampled_actions[start_idx + 1]};
        joystick joystick_canon{sampled_actions[start_idx + 2], sampled_actions[start_idx + 3]};
        button fire_button(sampled_actions[start_idx + 4] > 0);

        actions[i] = {joystick_direction, joystick_canon, fire_button, {false}, {false}};
    }

    return actions;
}
