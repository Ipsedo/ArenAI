//
// Created by samuel on 12/10/2025.
//

#include "../distributions/truncated_normal.h"

#include <arenai_core/constants.h>

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    torch::Tensor phi(const torch::Tensor &z) {
        return torch::exp(-0.5 * torch::pow(z, 2.0)) / std::sqrt(2.0 * M_PI);
    }

    torch::Tensor theta(const torch::Tensor &x) {
        return 0.5 * (1.0 + torch::erf(x / std::sqrt(2.0)));
    }

    torch::Tensor theta_inv(const torch::Tensor &theta) {
        return std::sqrt(2.0) * torch::erfinv(2.0 * theta - 1.0);
    }

    torch::Tensor truncated_normal_log_pdf(
        const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma,
        const float min_value, const float max_value) {

        const auto alpha = (min_value - mu) / sigma;
        const auto beta = (max_value - mu) / sigma;

        constexpr float z_eps = 1e-4f;
        const auto Z = torch::clamp_min(theta(beta) - theta(alpha), z_eps);

        return -0.5 * torch::pow((x - mu) / sigma, 2.0) - torch::log(sigma)
               - 0.5 * std::log(2.0 * M_PI) - torch::log(Z);
    }

    torch::Tensor truncated_normal_pdf(
        const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma,
        const float min_value, const float max_value) {

        const auto alpha = (min_value - mu) / sigma;
        const auto beta = (max_value - mu) / sigma;

        constexpr float z_eps = 1e-4f;
        const auto Z = torch::clamp_min(theta(beta) - theta(alpha), z_eps);

        return phi((x - mu) / sigma) / (Z * sigma);
    }

    torch::Tensor truncated_normal_sample(
        const torch::Tensor &mu, const torch::Tensor &sigma, const float min_value,
        const float max_value) {

        const auto alpha = (min_value - mu) / sigma;
        const auto beta = (max_value - mu) / sigma;

        constexpr float z_eps = 1e-4f;
        const auto Z = torch::clamp_min(theta(beta) - theta(alpha), z_eps);

        constexpr float cdf_eps = 1e-4f;
        const auto cdf =
            torch::clamp(theta(alpha) + torch::rand_like(mu) * Z, cdf_eps, 1.f - cdf_eps);

        return torch::clamp(theta_inv(cdf) * sigma + mu, min_value, max_value);
    }

    torch::Tensor truncated_normal_entropy(
        const torch::Tensor &mu, const torch::Tensor &sigma, const float min_value,
        const float max_value) {

        const auto alpha = (min_value - mu) / sigma;
        const auto beta = (max_value - mu) / sigma;

        constexpr float z_eps = 1e-4f;
        const auto Z = torch::clamp_min(theta(beta) - theta(alpha), z_eps);

        return 0.5 * torch::log(2.0 * M_PI * M_E * torch::pow(sigma, 2.0)) + torch::log(Z)
               + (alpha * phi(alpha) - beta * phi(beta)) / (2.0 * Z);
    }

    float truncated_normal_target_entropy(
        const int nb_actions, const float sigma, const float min_value, const float max_value) {
        return truncated_normal_entropy(
                   torch::zeros({nb_actions}), torch::ones({nb_actions}) * sigma, min_value,
                   max_value)
            .sum()
            .item<float>();
    }

}// namespace arenai::agent
