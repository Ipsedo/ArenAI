//
// Created by samuel on 12/10/2025.
//

#include "./truncated_normal.h"

#include <arenai_core/constants.h>

torch::Tensor phi(const torch::Tensor &z) {
    return torch::exp(-0.5 * torch::pow(z, 2.0)) / std::sqrt(2.0 * M_PI);
}

torch::Tensor theta(const torch::Tensor &x) { return 0.5 * (1.0 + torch::erf(x / std::sqrt(2.0))); }

torch::Tensor theta_inv(const torch::Tensor &theta) {
    return std::sqrt(2.0) * torch::erfinv(2.0 * theta - 1.0);
}

torch::Tensor truncated_normal_log_pdf(
    const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma,
    const float min_value, const float max_value) {

    const auto safe_sigma = torch::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto alpha =
        torch::clamp((min_value - mu) / safe_sigma, -ALPHA_BETA_BOUND, ALPHA_BETA_BOUND);
    const auto beta =
        torch::clamp((max_value - mu) / safe_sigma, -ALPHA_BETA_BOUND, ALPHA_BETA_BOUND);

    const auto z = theta(beta) - theta(alpha);

    return -0.5 * std::log(2.0 * M_PI) - torch::log(safe_sigma)
           - 0.5 * torch::pow((x - mu) / safe_sigma, 2.0) - torch::log(z);
}

torch::Tensor truncated_normal_sample(
    const torch::Tensor &mu, const torch::Tensor &sigma, const float min_value,
    const float max_value) {

    const auto safe_sigma = torch::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto alpha =
        torch::clamp((min_value - mu) / safe_sigma, -ALPHA_BETA_BOUND, ALPHA_BETA_BOUND);
    const auto beta =
        torch::clamp((max_value - mu) / safe_sigma, -ALPHA_BETA_BOUND, ALPHA_BETA_BOUND);

    const auto cdf = torch::clamp(
        theta(alpha)
            + at::rand(mu.sizes(), at::TensorOptions(mu.device())) * (theta(beta) - theta(alpha)),
        EPSILON, 1.f - EPSILON);

    return torch::clamp(theta_inv(cdf) * safe_sigma + mu, min_value, max_value);
}
