//
// Created by samuel on 18/09/2026.
//

#include "./bernoulli.h"

#include <cmath>

#include "../networks/constants.h"

using namespace arenai;
using namespace arenai::agent;

namespace {

    // float32: 1 - EPSILON (1e-8) rounds back to 1 and log(1 - p) would still
    // hit -inf when the sigmoid saturates, so the upper clamp needs its own margin
    constexpr float PROBA_MAX = 1.f - 1e-6f;

    torch::Tensor clamp_proba(const torch::Tensor &probabilities) {
        return torch::clamp(probabilities, agent::EPSILON, PROBA_MAX);
    }

}// namespace

namespace arenai::agent {

    torch::Tensor bernoulli_sample(const torch::Tensor &probabilities) {
        return torch::bernoulli(clamp_proba(probabilities));
    }

    torch::Tensor bernoulli_max_action(const torch::Tensor &probabilities) {
        return (probabilities > 0.5f).to(probabilities.dtype());
    }

    torch::Tensor
    bernoulli_log_proba(const torch::Tensor &actions, const torch::Tensor &probabilities) {
        const auto clamped_proba = clamp_proba(probabilities);
        return actions * torch::log(clamped_proba)
               + (1.f - actions) * torch::log(1.f - clamped_proba);
    }

    torch::Tensor bernoulli_entropy(const torch::Tensor &probabilities) {
        const auto clamped_proba = clamp_proba(probabilities);
        return -clamped_proba * torch::log(clamped_proba)
               - (1.f - clamped_proba) * torch::log(1.f - clamped_proba);
    }

    float bernoulli_maximum_entropy() { return std::log(2.f); }

}// namespace arenai::agent
