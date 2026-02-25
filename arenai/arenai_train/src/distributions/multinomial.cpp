//
// Created by samuel on 22/02/2026.
//

#include "./multinomial.h"

#include "arenai_core/constants.h"

torch::Tensor multinomial_sample(const torch::Tensor &probabilities) {
    const auto clamped_proba = torch::clamp(probabilities, EPSILON, 1.0);
    const auto idx = torch::multinomial(clamped_proba, 1, false);
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

float multinomial_target_entropy(const float &target_fire_probability) {
    const float no_fire_probability = 1.f - target_fire_probability;
    return -(
        target_fire_probability * std::log(target_fire_probability)
        + no_fire_probability * std::log(no_fire_probability));
}
