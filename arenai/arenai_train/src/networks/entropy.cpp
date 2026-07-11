//
// Created by samuel on 12/10/2025.
//

#include "./entropy.h"

#include <algorithm>
#include <cmath>
#include <numbers>

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

    TargetEntropyWarmup::TargetEntropyWarmup(
        const float initial_target_entropy, const float final_target_entropy, const int warmup_step)
        : initial(initial_target_entropy), final(final_target_entropy), warmup_step(warmup_step),
          current_step(register_buffer("current_step", torch::tensor({0L}))) {}

    void TargetEntropyWarmup::step() { current_step += 1; }

    torch::Tensor TargetEntropyWarmup::target_entropy() const {
        const float progress = std::min(
            1.f,
            static_cast<float>(current_step.item<int64_t>()) / static_cast<float>(warmup_step));
        const float cosine = 0.5f * (1.f - std::cos(std::numbers::pi_v<float> * progress));

        return torch::tensor(
            {initial + (final - initial) * cosine},
            torch::TensorOptions().device(current_step.device()));
    }

}// namespace arenai::train
