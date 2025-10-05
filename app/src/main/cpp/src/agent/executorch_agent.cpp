//
// Created by samuel on 03/10/2025.
//

#include "./executorch_agent.h"

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
    : actor_module(copy_asset_to_files(
        app->activity->assetManager, pte_asset_path.c_str(), get_cache_dir(app))) {}

std::vector<Action> ExecuTorchAgent::act(const std::vector<State> &state) { return {}; }
