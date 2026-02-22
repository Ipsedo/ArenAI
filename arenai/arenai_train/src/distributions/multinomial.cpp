//
// Created by samuel on 22/02/2026.
//

#include "./multinomial.h"

#include "arenai_core/constants.h"

torch::Tensor gumbel_hard(const torch::Tensor &probabilities) {
    const auto clamped_proba = torch::clamp(probabilities, EPSILON, 1.0);
    const auto idx = torch::argmax(clamped_proba, -1, true);
    const auto one_hot = torch::zeros_like(clamped_proba).scatter_(1, idx, 1.0);
    return (one_hot - clamped_proba).detach() + clamped_proba;
}

torch::Tensor
multinomial_log_proba(const torch::Tensor &action, const torch::Tensor &probabilities) {
    const auto clamped_proba = torch::clamp(probabilities, EPSILON, 1.0);
    const auto idx = action.argmax(-1).to(torch::kLong).unsqueeze(-1);
    const auto proba = torch::gather(clamped_proba, -1, idx);
    return torch::log(proba);
}

torch::Tensor multinomial_entropy(const torch::Tensor &probabilities) {
    const auto clamped_proba = torch::clamp(probabilities, EPSILON, 1.0);
    return -torch::sum(clamped_proba * torch::log(clamped_proba), -1, true);
}

float multinomial_target_entropy() { return -(0.2f * std::log(0.2f) + 0.8f * std::log(0.8f)); }
