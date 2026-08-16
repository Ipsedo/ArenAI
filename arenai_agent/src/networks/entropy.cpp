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

    AlphaParameter::AlphaParameter(const float initial_alpha)
        : log_alpha_tensor(
            register_parameter("log_alpha", torch::tensor({std::log(initial_alpha)}))) {}

    torch::Tensor AlphaParameter::log_alpha() { return log_alpha_tensor; }

    torch::Tensor AlphaParameter::alpha() { return log_alpha().exp(); }

    MultiAlphaParameters::MultiAlphaParameters(
        const float initial_alpha, const float min_alpha, const float max_alpha,
        const int nb_alphas)
        : log_alpha_tensor(register_parameter(
            "log_alpha",
            torch::full({nb_alphas}, std::log(std::clamp(initial_alpha, min_alpha, max_alpha))))),
          min_log_alpha(std::log(min_alpha)), max_log_alpha(std::log(max_alpha)) {}

    torch::Tensor MultiAlphaParameters::log_alpha() {
        {
            const torch::NoGradGuard no_grad;
            log_alpha_tensor.data().clamp_(min_log_alpha, max_log_alpha);
        }

        return log_alpha_tensor;
    }

    torch::Tensor MultiAlphaParameters::alpha() { return log_alpha().exp(); }

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

}// namespace arenai::agent
