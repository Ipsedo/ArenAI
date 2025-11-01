//
// Created by samuel on 03/10/2025.
//

#include "./executorch_agent.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>

#include <arenai_core/constants.h>

#include "./loader.h"

/*
 * Truncated normal sample
 */

float erfinv(const float x) {
    if (std::isnan(x)) return std::numeric_limits<float>::quiet_NaN();
    if (x == 1.0f) return std::numeric_limits<float>::infinity();
    if (x == -1.0f) return -std::numeric_limits<float>::infinity();

    const float eps = 10.0f * std::numeric_limits<float>::epsilon();
    float xc = std::clamp(x, -1.0f + eps, 1.0f - eps);

    constexpr float a = 0.147f;
    constexpr float pi = static_cast<float>(M_PI);
    const float s = (xc >= 0.0f) ? 1.0f : -1.0f;

    const float ln = std::log1p(-xc * xc);
    const float t = 2.0f / (pi * a) + 0.5f * ln;
    float y = s * std::sqrt(std::sqrt(t * t - ln / a) - t);

    const float c = 2.0f / std::sqrt(pi);
    for (int i = 0; i < 3; ++i) {
        float ey = std::erf(y);
        float dy = c * std::exp(-(y * y));
        y = y - (ey - xc) / dy;
    }

    return y;
}

float theta(const float x) { return 0.5f * (1.f + std::erf(x / std::sqrt(2.f))); }

float theta_inv(const float theta) { return std::sqrt(2.0f) * erfinv(2.0f * theta - 1.0f); }

float truncated_normal_sample(
    std::mt19937 rng, const float mu, const float sigma, const float min_value,
    const float max_value) {

    const auto safe_sigma = std::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto alpha =
        std::clamp((min_value - mu) / safe_sigma, -ALPHA_BETA_BOUND, ALPHA_BETA_BOUND);
    const auto beta =
        std::clamp((max_value - mu) / safe_sigma, -ALPHA_BETA_BOUND, ALPHA_BETA_BOUND);

    std::uniform_real_distribution<float> u_dist(0.f, 1.f);
    const auto cdf =
        std::clamp(theta(alpha) + u_dist(rng) * (theta(beta) - theta(alpha)), 0.f, 1.f);

    return std::clamp(theta_inv(cdf) * safe_sigma + mu, min_value, max_value);
}

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
    auto buffer_proprioception =
        std::vector<float>(static_cast<size_t>(N * ENEMY_PROPRIOCEPTION_SIZE));

    unsigned long idx_vision = 0;
    size_t idx_proprioception = 0;
    for (int64_t n = 0; n < N; n++) {
        const auto &[img, proprioception] = state[static_cast<size_t>(n)];
        for (int64_t c = 0; c < C; c++) {
            const auto &plane = img[static_cast<size_t>(c)];
            for (int64_t h = 0; h < H; h++) {
                const auto &row = plane[static_cast<size_t>(h)];
                for (int64_t w = 0; w < W; w++) {
                    // need convert it because LibTorch model doesn't export normalization [-1.0, 1.0]
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

    const auto dtype = torch::executor::ScalarType::Float;

    auto vision_tensor = executorch::extension::from_blob(
        static_cast<void *>(buffer_vision.data()), {static_cast<int>(N), C, H, W}, dtype);
    auto proprioception_tensor = executorch::extension::from_blob(
        static_cast<void *>(buffer_proprioception.data()),
        {static_cast<int>(N), ENEMY_PROPRIOCEPTION_SIZE}, dtype);

    auto output = actor_module.forward({vision_tensor, proprioception_tensor});

    if (!output.ok()) throw std::runtime_error(executorch::runtime::to_string(output.error()));

    auto mu = output->at(0).toTensor().const_data_ptr<float>();
    auto sigma = output->at(1).toTensor().const_data_ptr<float>();

    std::vector<Action> actions(N);
    for (int i = 0; i < N; i++) {
        std::vector<float> sampled_action(ENEMY_NB_ACTION);
        for (int a = 0; a < ENEMY_NB_ACTION; a++)
            sampled_action[a] = truncated_normal_sample(
                rng, mu[i * ENEMY_NB_ACTION + a], sigma[i * ENEMY_NB_ACTION + a], -1.f, 1.f);

        joystick joystick_direction{sampled_action[0], sampled_action[1]};
        joystick joystick_canon{sampled_action[2], sampled_action[3]};
        button fire_button(sampled_action[4] > 0);

        actions[i] = {joystick_direction, joystick_canon, fire_button};
    }

    return actions;
}
