#include "./ppo_objective.h"

#include <limits>

#include "../networks/constants.h"

namespace arenai::agent {

    namespace {
        constexpr float LOG_RATIO_MAX_ABS = 3.f;
    }// namespace

    torch::Tensor generalized_advantages(
        const torch::Tensor &deltas, const torch::Tensor &cuts, const float discount) {
        auto advantages = torch::zeros_like(deltas);
        auto running = torch::zeros_like(deltas[0]);
        for (int64_t t = deltas.size(0) - 1; t >= 0; t--) {
            running = deltas[t] + discount * (1.f - cuts[t]) * running;
            advantages[t] = running;
        }
        return advantages;
    }

    torch::Tensor normalized(const torch::Tensor &advantages, const torch::Tensor &valids) {
        const auto nb_valid = valids.sum().clamp_min(1.f);
        const auto mean = torch::sum(advantages * valids) / nb_valid;
        const auto std =
            torch::sqrt(torch::sum(torch::square(advantages - mean) * valids) / nb_valid);
        return (advantages - mean) / (std + EPSILON);
    }

    ClippedSurrogate clipped_surrogate(
        const torch::Tensor &log_probs, const torch::Tensor &old_log_probs,
        const torch::Tensor &advantages, const float clip_epsilon) {
        const auto log_ratio =
            torch::clamp(log_probs - old_log_probs, -LOG_RATIO_MAX_ABS, LOG_RATIO_MAX_ABS);
        const auto ratio = torch::exp(log_ratio);
        const auto clipped_ratio = torch::clamp(ratio, 1.f - clip_epsilon, 1.f + clip_epsilon);

        return {
            .surrogate = torch::min(ratio * advantages, clipped_ratio * advantages),
            .log_ratio = log_ratio,
            .ratio = ratio,
            .kl = (ratio - 1.f - log_ratio).detach().flatten()};
    }

    KlStop::KlStop(const float target_kl)
        : max_kl(target_kl > 0.f ? 1.5f * target_kl : std::numeric_limits<float>::infinity()),
          is_stopped(false), nb_planned_steps(0), nb_steps(0) {}

    bool KlStop::stopped() const { return is_stopped; }

    bool KlStop::stop_exceeded(const float kl) {
        is_stopped = is_stopped || kl > max_kl;
        return is_stopped;
    }

    void KlStop::on_planned_steps(const int nb_steps) { nb_planned_steps += nb_steps; }

    void KlStop::on_step() { nb_steps++; }

    float KlStop::skip_fraction() const {
        if (nb_planned_steps == 0) return 0.f;
        return 1.f - static_cast<float>(nb_steps) / static_cast<float>(nb_planned_steps);
    }

}// namespace arenai::agent
