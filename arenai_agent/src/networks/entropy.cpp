//
// Created by samuel on 12/10/2025.
//

#include "./entropy.h"

#include <algorithm>
#include <cmath>
#include <numbers>

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    AlphaParameters::AlphaParameters(const float initial_alpha, const int nb_alphas)
        : log_alpha(register_parameter(
            "log_alpha",
            torch::full({nb_alphas}, std::log(std::clamp(initial_alpha, MIN_ALPHA, MAX_ALPHA))))) {}

    torch::Tensor AlphaParameters::alpha() const { return torch::exp(log_alpha); }

    void AlphaParameters::clamp() const {
        const torch::NoGradGuard no_grad;
        log_alpha.clamp_(std::log(MIN_ALPHA), std::log(MAX_ALPHA));
    }

    torch::Tensor AlphaParameters::loss(const torch::Tensor &entropy_error) const {
        return torch::sum(log_alpha * entropy_error.detach());
    }

    /*
     * SAC
     */

    SacAlpha::SacAlpha(const float initial_alpha, const int nb_alphas)
        : log_alpha(
            register_parameter("log_alpha", torch::full({nb_alphas}, std::log(initial_alpha)))) {}

    torch::Tensor SacAlpha::alpha() const { return torch::exp(log_alpha); }

    /*
     * PID Lagrangian
     */

    PidLagrangianAlphaParameters::PidLagrangianAlphaParameters(
        const float k_p, const float k_i, const float k_d, const float initial_alpha,
        const int nb_alphas)
        : k_p(k_p), k_i(k_i), k_d(k_d),
          previous_entropy(register_buffer("previous_entropy", torch::zeros({1, nb_alphas}))),
          has_previous(register_buffer("has_previous", torch::zeros({1}))),
          integral(register_buffer(
              "integral", torch::full({1, nb_alphas}, std::clamp(initial_alpha, 0.f, MAX_ALPHA)))),
          alpha_tensor(register_buffer(
              "alpha", torch::full({1, nb_alphas}, std::clamp(initial_alpha, 0.f, MAX_ALPHA)))) {}

    torch::Tensor PidLagrangianAlphaParameters::alpha() const { return alpha_tensor; }

    void PidLagrangianAlphaParameters::update(
        const torch::Tensor &entropy, const torch::Tensor &target_entropy) const {
        const torch::NoGradGuard no_grad;

        const auto mean_entropy = torch::mean(entropy.detach(), 0, true).view_as(integral);
        const auto error =
            torch::mean(target_entropy.detach() - entropy.detach(), 0, true).view_as(integral);

        integral.copy_(torch::clamp(integral + k_i * error, 0.f, MAX_ALPHA));

        const auto derivative = has_previous * (previous_entropy - mean_entropy);

        previous_entropy.copy_(mean_entropy);
        has_previous.fill_(1.f);

        alpha_tensor.copy_(torch::clamp(k_p * error + integral + k_d * derivative, 0.f, MAX_ALPHA));
    }

}// namespace arenai::agent
