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

    const auto alpha = (min_value - mu) / safe_sigma;
    const auto beta = (max_value - mu) / safe_sigma;

    const auto Z = torch::clamp_min(theta(beta) - theta(alpha), EPSILON);

    return -0.5 * torch::pow((x - mu) / safe_sigma, 2.0) - torch::log(safe_sigma)
           - 0.5 * std::log(2.0 * M_PI) - torch::log(Z);
}

torch::Tensor truncated_normal_pdf(
    const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma,
    const float min_value, const float max_value) {

    const auto safe_sigma = torch::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto alpha = (min_value - mu) / safe_sigma;
    const auto beta = (max_value - mu) / safe_sigma;

    return phi((x - mu) / safe_sigma) / ((theta(beta) - theta(alpha)) * safe_sigma);
}

torch::Tensor truncated_normal_sample(
    const torch::Tensor &mu, const torch::Tensor &sigma, const float min_value,
    const float max_value) {

    const auto safe_sigma = torch::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto alpha = (min_value - mu) / safe_sigma;
    const auto beta = (max_value - mu) / safe_sigma;

    const auto Z = torch::clamp_min(theta(beta) - theta(alpha), EPSILON);

    const auto cdf = theta(alpha) + at::rand(mu.sizes(), at::TensorOptions(mu.device())) * Z;

    return theta_inv(cdf) * safe_sigma + mu;
}

torch::Tensor truncated_normal_entropy(
    const torch::Tensor &mu, const torch::Tensor &sigma, const float min_value,
    const float max_value) {

    const auto safe_sigma = torch::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto alpha = (min_value - mu) / safe_sigma;
    const auto beta = (max_value - mu) / safe_sigma;

    const auto Z = torch::clamp_min(theta(beta) - theta(alpha), EPSILON);

    return 0.5 * torch::log(2.0 * M_PI * M_E * torch::pow(safe_sigma, 2.0)) + torch::log(Z)
           + (alpha * phi(alpha) - beta * phi(beta)) / (2.0 * Z);
}

float truncated_normal_target_entropy(
    const int nb_actions, const float min_value, const float max_value) {
    return truncated_normal_entropy(
               torch::zeros({nb_actions}), torch::ones({nb_actions}) * std::sqrt(2.0), min_value,
               max_value)
        .sum()
        .item<float>();
}

torch::Tensor gaussian_tanh_sample(const torch::Tensor &mu, const torch::Tensor &sigma) {
    const auto safe_sigma = torch::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto eps = torch::randn_like(mu);

    const auto u = mu + safe_sigma * eps;
    return torch::tanh(u);
}

torch::Tensor
gaussian_tanh_log_pdf(const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma) {
    const auto safe_sigma = torch::clamp(sigma, SIGMA_MIN, SIGMA_MAX);

    const auto u = 0.5 * torch::log((1.0 + x + EPSILON) / (1.0 - x + EPSILON));

    const auto log_unnormalized = -0.5 * torch::pow((u - mu) / safe_sigma, 2.0);
    const auto log_normalization = torch::log(safe_sigma) + 0.5 * std::log(2.0 * M_PI);

    const auto log_gaussian = log_unnormalized - log_normalization;
    const auto log_det_jacobian = torch::log(1.0 - x.pow(2.0) + EPSILON);

    return log_gaussian - log_det_jacobian;
}
