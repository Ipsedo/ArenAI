#include "../distributions/gaussian_tanh.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    namespace {
        torch::Tensor log_tanh_derivative(const torch::Tensor &u) {
            return 2.0 * (std::log(2.0) - u - torch::softplus(-2.0 * u));
        }
    }// namespace

    std::pair<torch::Tensor, torch::Tensor>
    gaussian_tanh_sample(const torch::Tensor &mu, const torch::Tensor &sigma) {
        const auto u = mu + sigma * torch::randn_like(mu);
        return {torch::tanh(u), u};
    }

    torch::Tensor gaussian_tanh_mode(const torch::Tensor &mu) { return torch::tanh(mu); }

    torch::Tensor gaussian_tanh_log_pdf(
        const torch::Tensor &u, const torch::Tensor &mu, const torch::Tensor &sigma) {
        const auto log_unnormalized = -0.5 * torch::pow((u - mu) / sigma, 2);
        const auto log_normalization = torch::log(sigma) + 0.5 * std::log(2.0 * M_PI);

        return log_unnormalized - log_normalization - log_tanh_derivative(u);
    }

    torch::Tensor gaussian_tanh_entropy(const torch::Tensor &mu, const torch::Tensor &sigma) {
        const auto u = mu + sigma * torch::randn_like(mu);
        return -gaussian_tanh_log_pdf(u, mu, sigma);
    }

    torch::Tensor gaussian_tanh_kl(
        const torch::Tensor &mu_p, const torch::Tensor &sigma_p, const torch::Tensor &mu_q,
        const torch::Tensor &sigma_q) {
        return torch::log(sigma_q / sigma_p)
               + (torch::pow(sigma_p, 2) + torch::pow(mu_p - mu_q, 2))
                     / (2.0 * torch::pow(sigma_q, 2))
               - 0.5;
    }

    float gaussian_tanh_target_entropy(
        const int nb_actions, const float target_sigma, const int nb_samples) {
        const auto log_gauss = 0.5 * std::log(2.0 * M_PI * M_E * target_sigma * target_sigma);

        const auto u = torch::randn({nb_samples}) * target_sigma;
        const auto mean_log_det = log_tanh_derivative(u).mean().item<float>();

        const auto h_per_dim = log_gauss + mean_log_det;
        return static_cast<float>(nb_actions * h_per_dim);
    }

}// namespace arenai::agent
