//
// Created by samuel on 08/02/2026.
//

#include "./beta_law.h"

#include <arenai_core/constants.h>

// Kumaraswamy

inline torch::Tensor clamp_pos(const torch::Tensor &t) { return torch::clamp(t, EPSILON); }

inline torch::Tensor clamp_unit_open(const torch::Tensor &x) {
    return torch::clamp(x, EPSILON, 1.0 - EPSILON);
}

torch::Tensor beta_law_sample(const torch::Tensor &alpha, const torch::Tensor &beta) {
    const auto clamped_alpha = clamp_pos(alpha);
    const auto clamped_beta = clamp_pos(beta);

    const auto u =
        torch::clamp(torch::rand_like(clamped_alpha + clamped_beta), EPSILON, 1.0 - EPSILON);
    const auto inner = torch::clamp(1.0 - torch::pow(1.0 - u, 1.0 / clamped_beta), EPSILON, 1.0);

    return clamp_unit_open(torch::pow(inner, 1.0 / clamped_alpha)) * 2.f - 1.f;
}

torch::Tensor
beta_law_log_proba(const torch::Tensor &x, const torch::Tensor &alpha, const torch::Tensor &beta) {
    const auto clamped_alpha = torch::clamp_min(alpha, EPSILON);
    const auto clamped_beta = torch::clamp_min(beta, EPSILON);

    const auto y = torch::clamp((x + 1.f) / 2.f, EPSILON, 1.f - EPSILON);
    const auto ya = torch::clamp(torch::pow(y, clamped_alpha), EPSILON, 1.f - EPSILON);

    const auto logp = torch::log(clamped_alpha) + torch::log(clamped_beta)
                      + (clamped_alpha - 1.0) * torch::log(y)
                      + (clamped_beta - 1.0) * torch::log1p(-ya);

    return logp - std::log(2.0);
}

torch::Tensor beta_law_entropy(const torch::Tensor &alpha, const torch::Tensor &beta) {
    const auto clamped_alpha = clamp_pos(alpha);
    const auto clamped_beta = clamp_pos(beta);

    // Euler–Mascheroni constant
    constexpr double EULER_GAMMA = 0.57721566490153286060651209;

    const auto H_b = torch::digamma(clamped_beta + 1.0) + EULER_GAMMA;// harmonic number extension
    return 1.0 - 1.0 / clamped_alpha + (1.0 - 1.0 / clamped_beta) * H_b
           - torch::log(clamped_alpha * clamped_beta);
}

float beta_law_target_entropy(const int &nb_actions) {
    return beta_law_entropy(torch::tensor(1.f), torch::tensor(1.f)).item<float>() * nb_actions;
}
