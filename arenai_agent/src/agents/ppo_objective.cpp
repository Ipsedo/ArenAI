#include "./ppo_objective.h"

#include "../networks/constants.h"

namespace arenai::agent {

    namespace {
        constexpr float LOG_RATIO_MAX_ABS = 3.f;

        torch::Tensor group_surrogate(
            const torch::Tensor &log_ratio, const torch::Tensor &advantages,
            const float clip_epsilon) {
            const auto ratio = torch::exp(log_ratio);
            const auto clipped_ratio = torch::clamp(ratio, 1.f - clip_epsilon, 1.f + clip_epsilon);
            return torch::min(ratio * advantages, clipped_ratio * advantages);
        }
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

    torch::Tensor aim_advantages(
        const torch::Tensor &aim_rewards, const torch::Tensor &cuts, const torch::Tensor &valids,
        const float gamma, const float lambda) {
        return normalized(generalized_advantages(aim_rewards, cuts, gamma * lambda), valids);
    }

    PolicyLogProbs PolicyLogProbs::split(
        const torch::Tensor &continuous_log_probs, const torch::Tensor &discrete_log_probs) {
        return {
            .aim = continuous_log_probs.slice(-1, FIRST_AIM_ACTION).sum(-1, true),
            .rest = continuous_log_probs.slice(-1, 0, FIRST_AIM_ACTION).sum(-1, true)
                    + discrete_log_probs};
    }

    ClippedSurrogate clipped_surrogate(
        const PolicyLogProbs &current, const PolicyLogProbs &old,
        const torch::Tensor &aim_advantages, const torch::Tensor &advantages,
        const float clip_epsilon) {
        const auto aim_log_ratio =
            torch::clamp(current.aim - old.aim, -LOG_RATIO_MAX_ABS, LOG_RATIO_MAX_ABS);
        const auto rest_log_ratio =
            torch::clamp(current.rest - old.rest, -LOG_RATIO_MAX_ABS, LOG_RATIO_MAX_ABS);

        const auto log_ratio = aim_log_ratio + rest_log_ratio;

        return {
            .surrogate = group_surrogate(aim_log_ratio, aim_advantages, clip_epsilon)
                         + group_surrogate(rest_log_ratio, advantages, clip_epsilon),
            .log_ratio = log_ratio,
            .ratio = torch::exp(log_ratio)};
    }

}// namespace arenai::agent
