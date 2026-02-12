//
// Created by samuel on 10/02/2026.
//

#include "../distributions/gaussian_tanh.h"

#include <arenai_core/constants.h>

torch::Tensor gaussian_tanh_sample(const torch::Tensor &mu, const torch::Tensor &sigma) {
    const auto safe_sigma = torch::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto noise = torch::randn_like(mu);

    const auto u = mu + safe_sigma * noise;
    return torch::tanh(u);
}

torch::Tensor
gaussian_tanh_log_pdf(const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma) {
    const auto safe_sigma = torch::clamp(sigma, SIGMA_MIN, SIGMA_MAX);
    const auto a = torch::clamp(x, -1 + EPSILON, 1 - EPSILON);

    const auto z = atanh(a);

    const auto log_unnormalized = -0.5 * torch::pow((z - mu) / safe_sigma, 2);
    const auto log_normalization = torch::log(safe_sigma) + 0.5 * std::log(2.0 * M_PI);
    const auto log_gauss = log_unnormalized - log_normalization;

    const auto log_det = torch::log(1.0 - a.pow(2) + EPSILON);

    return log_gauss - log_det;
}
