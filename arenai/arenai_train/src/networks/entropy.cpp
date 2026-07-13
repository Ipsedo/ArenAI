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
using namespace arenai::train;

namespace arenai::train {

    AlphaParameter::AlphaParameter(const float initial_alpha)
        : log_alpha_tensor(
            register_parameter("log_alpha", torch::tensor({std::log(initial_alpha)}))) {}

    torch::Tensor AlphaParameter::log_alpha() { return log_alpha_tensor; }

    torch::Tensor AlphaParameter::alpha() { return log_alpha().exp(); }

    /*
     * Target entropy
     */

    AbstractTargetEntropyWarmup::AbstractTargetEntropyWarmup(
        const float initial_value, const float final_value, const int warmup_step)
        : initial(initial_value), final(final_value), warmup_step(warmup_step),
          current_step(register_buffer("current_step", torch::tensor({0L}))) {}

    void AbstractTargetEntropyWarmup::step() { current_step += 1; }

    torch::Tensor AbstractTargetEntropyWarmup::target_entropy() {
        const float progress = std::min(
            1.f,
            static_cast<float>(current_step.item<int64_t>()) / static_cast<float>(warmup_step));
        const float cosine = 0.5f * (1.f - std::cos(std::numbers::pi_v<float> * progress));

        return torch::tensor(
            {to_target_entropy(initial + (final - initial) * cosine)},
            torch::TensorOptions().device(current_step.device()));
    }

    /*
     * Discrete
     */

    DiscreteTargetEntropyWarmup::DiscreteTargetEntropyWarmup(
        const int nb_actions, const float initial_factor, const float final_factor,
        const int warmup_step)
        : AbstractTargetEntropyWarmup(initial_factor, final_factor, warmup_step),
          nb_actions(nb_actions) {}

    float DiscreteTargetEntropyWarmup::to_target_entropy(const float value) {
        return multinomial_maximum_entropy(nb_actions) * value;
    }

    /*
     * Continuous
     */

    ContinuousTargetEntropyWarmup::ContinuousTargetEntropyWarmup(
        const int nb_actions, const float initial_sigma, const float final_sigma,
        const int warmup_step)
        : AbstractTargetEntropyWarmup(initial_sigma, final_sigma, warmup_step),
          nb_actions(nb_actions) {}

    float ContinuousTargetEntropyWarmup::to_target_entropy(const float value) {
        return truncated_normal_target_entropy(nb_actions, value);
    }

}// namespace arenai::train
