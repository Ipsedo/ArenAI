//
// Created by samuel on 03/10/2025.
//

#include "./executorch_agent.h"

#include <boost/math/special_functions/erf.hpp>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>

#include <arenai_core/constants.h>

#include "./loader.h"

/*
 * Truncated normal sample
 */

float theta(const float x) { return 0.5f * (1.f + std::erf(x / std::sqrt(2.f))); }

float theta_inv(const float theta) {
    return std::sqrt(2.0f) * boost::math::erf_inv(2.0f * theta - 1.0f);
}

float truncated_normal_sample(
    std::mt19937 &rng, const float mu, const float sigma, const float min_value,
    const float max_value) {

    const auto safe_sigma = std::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto alpha = (min_value - mu) / safe_sigma;
    const auto beta = (max_value - mu) / safe_sigma;

    const auto Z = std::max(theta(beta) - theta(alpha), EPSILON);

    std::uniform_real_distribution<float> u_dist(0.f, 1.f);
    const auto cdf = theta(alpha) + u_dist(rng) * Z;

    return theta_inv(cdf) * safe_sigma + mu;
}

/*
 * Gaussian tanh
 */

float gaussian_tanh_sample(std::mt19937 &rng, const float mu, const float sigma) {
    const auto safe_sigma = std::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    std::uniform_real_distribution<float> u_dist(0.f, 1.f);
    const auto noise = u_dist(rng);

    const auto u = mu + safe_sigma * noise;
    return std::tanh(u);
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
    const int64_t H = ENEMY_VISION_HEIGHT;
    const int64_t W = ENEMY_VISION_WIDTH;

    const auto vision_elems_per_sample = static_cast<size_t>(C * H * W);
    const auto proprio_elems_per_sample = static_cast<size_t>(ENEMY_PROPRIOCEPTION_SIZE);

    auto buffer_vision = std::vector<float>(static_cast<size_t>(N * C * H * W));
    auto buffer_proprioception =
        std::vector<float>(static_cast<size_t>(N * ENEMY_PROPRIOCEPTION_SIZE));

    size_t vision_off = 0;
    size_t proprio_off = 0;

    for (int64_t n = 0; n < N; ++n) {
        const auto &[img, proprioception] = state[static_cast<size_t>(n)];

        const uint8_t *src = img.pixels.data();
        float *dst = buffer_vision.data() + vision_off;

        for (size_t k = 0; k < vision_elems_per_sample; ++k)
            dst[k] = 2.f * static_cast<float>(src[k]) / 255.0f - 1.f;

        vision_off += vision_elems_per_sample;

        std::memcpy(
            buffer_proprioception.data() + proprio_off, proprioception.data(),
            sizeof(float) * proprio_elems_per_sample);
        proprio_off += proprio_elems_per_sample;
    }

    const auto dtype = torch::executor::ScalarType::Float;

    auto vision_tensor = executorch::extension::from_blob(
        static_cast<void *>(buffer_vision.data()),
        {static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W)},
        dtype);

    auto proprioception_tensor = executorch::extension::from_blob(
        static_cast<void *>(buffer_proprioception.data()),
        {static_cast<int>(N), static_cast<int>(ENEMY_PROPRIOCEPTION_SIZE)}, dtype);

    auto output = actor_module.forward({vision_tensor, proprioception_tensor});

    if (!output.ok()) throw std::runtime_error(executorch::runtime::to_string(output.error()));

    auto mu = output->at(0).toTensor().const_data_ptr<float>();
    auto sigma = output->at(1).toTensor().const_data_ptr<float>();
    auto discrete = output->at(2).toTensor().const_data_ptr<float>();

    std::vector<Action> actions(N);
    for (int i = 0; i < N; i++) {
        std::vector<float> continuous_action(ENEMY_NB_CONTINUOUS_ACTION);
        for (int a = 0; a < ENEMY_NB_CONTINUOUS_ACTION; a++)
            continuous_action[a] = truncated_normal_sample(
                rng, mu[i * ENEMY_NB_CONTINUOUS_ACTION + a],
                sigma[i * ENEMY_NB_CONTINUOUS_ACTION + a], -1.f, 1.f);

        std::discrete_distribution<int> discrete_dist(
            {discrete[i * ENEMY_NB_DISCRETE_ACTION], discrete[i * ENEMY_NB_DISCRETE_ACTION + 1]});

        joystick joystick_direction{continuous_action[0], continuous_action[1]};
        joystick joystick_canon{continuous_action[2], continuous_action[3]};
        button fire_button(discrete_dist(rng) == 0);

        actions[i] = {joystick_direction, joystick_canon, fire_button};
    }

    return actions;
}
