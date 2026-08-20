//
// Created by samuel on 12/10/2025.
//

#include "./entropy.h"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "../distributions/multinomial.h"
#include "../distributions/truncated_normal.h"

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
     * Constant target entropy
     */

    ConstantTargetEntropy::ConstantTargetEntropy(const float initial_target)
        : initial_target(register_buffer("initial_target", torch::tensor(initial_target))) {}

    torch::Tensor ConstantTargetEntropy::target_entropy() { return initial_target; }

    ConstantDiscreteTargetEntropy::ConstantDiscreteTargetEntropy(const float fire_probability)
        : ConstantTargetEntropy(multinomial_target_entropy(fire_probability)) {}

    ConstantContinuousTargetEntropy::ConstantContinuousTargetEntropy(
        const int nb_continuous_action, const float target_sigma)
        : ConstantTargetEntropy(
            truncated_normal_target_entropy(nb_continuous_action, target_sigma)) {}

    ConstantContinuousPerActionTargetEntropy::ConstantContinuousPerActionTargetEntropy(
        const float target_sigma)
        : ConstantTargetEntropy(truncated_normal_target_entropy(1, target_sigma)) {}

    /*
     * Target entropy warmup
     */

    AbstractCosineAnnealingTargetEntropy::AbstractCosineAnnealingTargetEntropy(
        const float initial_value, const float final_value, const int warmup_step)
        : initial(initial_value), final(final_value), warmup_step(warmup_step),
          current_step(register_buffer(
              "current_step", torch::zeros({1}, torch::TensorOptions().dtype(torch::kLong)))) {}

    torch::Tensor AbstractCosineAnnealingTargetEntropy::target_entropy() {
        const float progress = std::min(
            1.f,
            static_cast<float>(current_step.item<int64_t>()) / static_cast<float>(warmup_step));
        const float cosine = 0.5f * (1.f - std::cos(std::numbers::pi_v<float> * progress));

        current_step += 1;

        return torch::tensor(
            {to_target_entropy(initial + (final - initial) * cosine)},
            torch::TensorOptions().device(current_step.device()));
    }

    /*
     * Discrete
     */

    DiscreteCosineAnnealingTargetEntropy::DiscreteCosineAnnealingTargetEntropy(
        const float initial_probability, const float final_probability, const int warmup_step)
        : AbstractCosineAnnealingTargetEntropy(
            initial_probability, final_probability, warmup_step) {}

    float DiscreteCosineAnnealingTargetEntropy::to_target_entropy(const float value) {
        return multinomial_target_entropy(value);
    }

    /*
     * Continuous
     */

    ContinuousCosineAnnealingTargetEntropy::ContinuousCosineAnnealingTargetEntropy(
        const int nb_actions, const float initial_sigma, const float final_sigma,
        const int warmup_step)
        : AbstractCosineAnnealingTargetEntropy(initial_sigma, final_sigma, warmup_step),
          nb_actions(nb_actions) {}

    float ContinuousCosineAnnealingTargetEntropy::to_target_entropy(const float value) {
        return truncated_normal_target_entropy(nb_actions, value);
    }

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
              "integral",
              torch::full(
                  {1, nb_alphas}, std::log(std::clamp(initial_alpha, MIN_ALPHA, MAX_ALPHA))))),
          log_alpha_tensor(register_buffer(
              "log_alpha",
              torch::full(
                  {1, nb_alphas}, std::log(std::clamp(initial_alpha, MIN_ALPHA, MAX_ALPHA))))) {}

    torch::Tensor PidLagrangianAlphaParameters::alpha() const { return log_alpha_tensor.exp(); }

    void PidLagrangianAlphaParameters::update(
        const torch::Tensor &entropy, const torch::Tensor &target_entropy) const {
        const torch::NoGradGuard no_grad;

        const auto mean_entropy = torch::mean(entropy.detach(), 0, true).view_as(integral);
        const auto error =
            torch::mean(target_entropy.detach() - entropy.detach(), 0, true).view_as(integral);

        integral.copy_(
            torch::clamp(integral + k_i * error, std::log(MIN_ALPHA), std::log(MAX_ALPHA)));

        const auto derivative =
            has_previous * torch::clamp_min(previous_entropy - mean_entropy, 0.f);

        previous_entropy.copy_(mean_entropy);
        has_previous.fill_(1.f);

        log_alpha_tensor.copy_(k_p * error + integral + k_d * derivative);
    }

}// namespace arenai::agent
