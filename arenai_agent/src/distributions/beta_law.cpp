//
// Created by samuel on 08/02/2026.
//

#include "../distributions/beta_law.h"

#include "../networks/constants.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    namespace {
        torch::Tensor to_alpha(const torch::Tensor &mu, const torch::Tensor &kappa) {
            return mu * kappa * KAPPA_SCALE;
        }

        torch::Tensor to_beta(const torch::Tensor &mu, const torch::Tensor &kappa) {
            return (1.f - mu) * kappa * KAPPA_SCALE;
        }
    }// namespace

    // the log-proba and entropy carry the log(2) change of scale from [0, 1] to [-1, 1]

    static torch::Tensor clamp_pos(const torch::Tensor &t) { return torch::clamp_min(t, EPSILON); }

    static torch::Tensor log_beta_function(const torch::Tensor &alpha, const torch::Tensor &beta) {
        return torch::lgamma(alpha) + torch::lgamma(beta) - torch::lgamma(alpha + beta);
    }

    torch::Tensor beta_law_sample(const torch::Tensor &mu, const torch::Tensor &kappa) {
        const auto alpha = to_alpha(mu, kappa);
        const auto beta = to_beta(mu, kappa);

        // Beta(α, β) = X / (X + Y) with X ~ Gamma(α, 1) and Y ~ Gamma(β, 1),
        // differentiable w.r.t. α and β through the implicit gradients of _standard_gamma
        const auto x = at::_standard_gamma(clamp_pos(alpha));
        const auto y = at::_standard_gamma(clamp_pos(beta));

        const auto u = torch::clamp(x / clamp_pos(x + y), EPSILON, 1.0 - EPSILON);
        return u * 2.f - 1.f;
    }

    torch::Tensor beta_law_log_proba(
        const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &kappa) {
        const auto alpha = to_alpha(mu, kappa);
        const auto beta = to_beta(mu, kappa);

        const auto clamped_alpha = clamp_pos(alpha);
        const auto clamped_beta = clamp_pos(beta);

        constexpr float eps = 1e-6f;
        const auto y = torch::clamp((x + 1.f) / 2.f, eps, 1.f - eps);

        return (clamped_alpha - 1.0) * torch::log(y) + (clamped_beta - 1.0) * torch::log1p(-y)
               - log_beta_function(clamped_alpha, clamped_beta) - std::log(2.0);
    }

    torch::Tensor beta_law_entropy(const torch::Tensor &mu, const torch::Tensor &kappa) {
        const auto alpha = to_alpha(mu, kappa);
        const auto beta = to_beta(mu, kappa);

        const auto clamped_alpha = clamp_pos(alpha);
        const auto clamped_beta = clamp_pos(beta);

        return log_beta_function(clamped_alpha, clamped_beta)
               - (clamped_alpha - 1.0) * torch::digamma(clamped_alpha)
               - (clamped_beta - 1.0) * torch::digamma(clamped_beta)
               + (clamped_alpha + clamped_beta - 2.0) * torch::digamma(clamped_alpha + clamped_beta)
               + std::log(2.0);
    }

    torch::Tensor beta_law_mean_action(const torch::Tensor &mu, const torch::Tensor &kappa) {
        const auto alpha = to_alpha(mu, kappa);
        const auto beta = to_beta(mu, kappa);

        const auto clamped_alpha = clamp_pos(alpha);
        const auto clamped_beta = clamp_pos(beta);

        return 2.f * clamped_alpha / (clamped_alpha + clamped_beta) - 1.f;
    }

    float beta_law_target_entropy(const int &nb_actions, const float target_concentration) {
        return beta_law_entropy(torch::tensor(0.5f), torch::tensor(target_concentration))
                   .item<float>()
               * static_cast<float>(nb_actions);
    }

}// namespace arenai::agent
