//
// Created by samuel on 03/10/2025.
//

#include "./executorch_agent.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include "./loader.h"

#define SIGMA_MIN 1e-6f
#define SIGMA_MAX 1e6f
#define ALPHA_BETA_BOUND 5.f

/*
 * truncated normal
 */

float erfinv(float x) {
    if (x <= -1.f) return -std::numeric_limits<float>::infinity();
    if (x >= 1.f) return std::numeric_limits<float>::infinity();
    if (x == 0.f) return 0.f;

    const float a = 0.147f;
    float ln = std::log(1.f - x * x);
    float term1 = 2.f / (static_cast<float>(M_PI) * a) + ln / 2.f;
    float term2 = ln / a;
    float sign = (x > 0) ? 1.f : -1.f;

    float initial = sign * std::sqrt(std::sqrt(term1 * term1 - term2) - term1);

    float y = initial;
    for (int i = 0; i < 2; i++) {
        float err = std::erf(y) - x;
        float deriv = (2.f / std::sqrt(static_cast<float>(M_PI))) * std::exp(-y * y);
        y -= err / deriv;
    }
    return y;
}

float phi(float z) {
    return std::exp(-0.5f * std::pow(z, 2.0f)) / std::sqrt(2.0f * static_cast<float>(M_PI));
}

float theta(float x) { return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f))); }

float theta_inv(float theta) { return std::sqrt(2.0f) * erfinv(2.0f * theta - 1.0f); }

float truncated_normal_sample(
    std::random_device rng, std::uniform_real_distribution<float> dist, float mu, float sigma,
    float min_value, float max_value) {

    const auto safe_sigma = std::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto alpha =
        std::clamp((min_value - mu) / safe_sigma, -ALPHA_BETA_BOUND, ALPHA_BETA_BOUND);
    const auto beta =
        std::clamp((max_value - mu) / safe_sigma, -ALPHA_BETA_BOUND, ALPHA_BETA_BOUND);

    const auto cdf = std::clamp(theta(alpha) + dist(rng) * (theta(beta) - theta(alpha)), 0.f, 1.f);

    return std::clamp(theta_inv(cdf) * safe_sigma + mu, min_value, max_value);
}

/*
 * ExecuTorch
 */

ExecuTorchAgent::ExecuTorchAgent(android_app *app, const std::string &pte_asset_path)
    : actor_module(
        copy_asset_to_files(app->activity->assetManager, pte_asset_path, get_cache_dir(app))) {}

std::vector<Action> ExecuTorchAgent::act(const std::vector<State> &state) {
    std::vector<float> visions(state.size() * 3 * ENEMY_VISION_SIZE * ENEMY_VISION_SIZE);
    std::vector<float> proprioception(state.size() * ENEMY_PROPRIOCEPTION_SIZE);

    int64_t idx_v = 0;
    int64_t idx_p = 0;

    for (const auto &s: state) {
        for (int64_t c = 0; c < 3; c++) {
            for (int64_t h = 0; h < ENEMY_VISION_SIZE; h++) {
                for (int64_t w = 0; w < ENEMY_VISION_SIZE; w++) {
                    visions[idx_v++] = 2.f * static_cast<float>(s.vision[h][w][c]) / 255.f - 1.f;
                }
            }
        }

        for (int f = 0; f < ENEMY_PROPRIOCEPTION_SIZE; f++)
            proprioception[idx_p++] = s.proprioception[f];
    }

    auto vision_tensor = executorch::extension::from_blob(
        visions.data(), {static_cast<int>(state.size()), 3, ENEMY_VISION_SIZE, ENEMY_VISION_SIZE});
    auto proprioception_tensor = executorch::extension::from_blob(
        proprioception.data(), {static_cast<int>(state.size()), ENEMY_PROPRIOCEPTION_SIZE});

    auto output = actor_module.forward({vision_tensor, proprioception_tensor});

    return {};
}
