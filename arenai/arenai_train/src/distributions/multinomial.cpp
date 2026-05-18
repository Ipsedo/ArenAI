//
// Created by samuel on 22/02/2026.
//

#include "./multinomial.h"

#include "arenai_core/constants.h"

torch::Tensor multinomial_sample(const torch::Tensor &probabilities) {
    const auto clamped_proba = torch::clamp(probabilities, EPSILON, 1.0 - EPSILON);
    const auto idx = torch::multinomial(clamped_proba, 1, false);
    const auto one_hot = torch::zeros_like(clamped_proba).scatter_(1, idx, 1.0);
    return one_hot;
}

torch::Tensor multinomial_entropy(const torch::Tensor &probabilities) {
    const auto clamped_proba = torch::clamp(probabilities, EPSILON, 1.0 - EPSILON);
    return -torch::sum(clamped_proba * torch::log(clamped_proba), -1, true);
}

float multinomial_target_entropy(const int &nb_actions, const float &factor) {
    return factor
           * multinomial_entropy(torch::ones({nb_actions}) / static_cast<float>(nb_actions))
                 .item<float>();
}

float multinomial_target_entropy(const float &shoot_probability) {
    return multinomial_entropy(torch::tensor({shoot_probability, 1.f - shoot_probability}))
        .item<float>();
}
