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

    /*
     * Alpha parameters [0; +inf[
     */

    AlphaParameters::AlphaParameters(const float initial_alpha, const int nb_alphas)
        : log_alpha_tensor(register_parameter(
            "log_alpha",
            torch::tensor(std::vector(nb_alphas, std::log(initial_alpha))).unsqueeze(0))) {}

    torch::Tensor AlphaParameters::log_alpha() { return log_alpha_tensor; }

    torch::Tensor AlphaParameters::alpha() { return log_alpha().exp(); }

    /*
     * Clamped alpha parameters
     */

    ClampedAlphaParameters::ClampedAlphaParameters(
        const float initial_alpha, const float min_alpha, const float max_alpha,
        const int nb_alphas)
        : AlphaParameters(std::clamp(initial_alpha, min_alpha, max_alpha), nb_alphas),
          min_log_alpha(std::log(min_alpha)), max_log_alpha(std::log(max_alpha)) {}

    torch::Tensor ClampedAlphaParameters::log_alpha() {
        auto curr_log_alpha = AlphaParameters::log_alpha();

        {
            const torch::NoGradGuard no_grad;
            curr_log_alpha.data().clamp_(min_log_alpha, max_log_alpha);
        }

        return curr_log_alpha;
    }

    /*
     * Abstract target entropy
     */

    // a schedule-less target ignores the progression
    void AbstractTargetEntropy::step(int64_t) {}

    /*
     * Constant target entropy
     */

    ConstantTargetEntropy::ConstantTargetEntropy(const float initial_target)
        : initial_target(register_buffer("initial_target", torch::tensor(initial_target))) {}

    torch::Tensor ConstantTargetEntropy::target_entropy() const { return initial_target; }

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
