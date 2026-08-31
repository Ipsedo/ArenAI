//
// Created by samuel on 31/08/2026.
//

#include "./warmup.h"

CosineAnnealing::CosineAnnealing(
    const float initial_value, const float final_value, const int64_t warmup_env_step)
    : initial(initial_value), final(final_value),
      warmup_env_step(std::max<int64_t>(1, warmup_env_step)), current_step(0) {}

float CosineAnnealing::value() const {
    const float progress =
        std::min(1.f, static_cast<float>(current_step) / static_cast<float>(warmup_env_step));
    const float cosine = 0.5f * (1.f - std::cos(std::numbers::pi_v<float> * progress));

    return initial + (final - initial) * cosine;
}

void CosineAnnealing::step(const int64_t nb_env_steps) { current_step += nb_env_steps; }
