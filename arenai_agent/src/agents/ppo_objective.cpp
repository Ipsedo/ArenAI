#include "./ppo_objective.h"

#include <limits>

#include "../networks/constants.h"
#include "./update_utils.h"

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

        // [N, 1] -> [N]
        torch::Tensor row_kl(const torch::Tensor &log_ratio) {
            return (torch::exp(log_ratio) - 1.f - log_ratio).detach().flatten();
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

    GroupKls trimmed_group_kls(const std::vector<RowKls> &row_kls) {
        std::vector<torch::Tensor> joints, aims, rests;
        for (const auto &[joint, aim, rest]: row_kls) {
            joints.push_back(joint);
            aims.push_back(aim);
            rests.push_back(rest);
        }
        return {.joint = trimmed_kl(joints), .aim = trimmed_kl(aims), .rest = trimmed_kl(rests)};
    }

    ClippedSurrogate clipped_surrogate(
        const PolicyLogProbs &current, const PolicyLogProbs &old,
        const torch::Tensor &aim_advantages, const torch::Tensor &advantages,
        const float clip_epsilon, const ActiveGroups &active) {
        const auto aim_log_ratio =
            torch::clamp(current.aim - old.aim, -LOG_RATIO_MAX_ABS, LOG_RATIO_MAX_ABS);
        const auto rest_log_ratio =
            torch::clamp(current.rest - old.rest, -LOG_RATIO_MAX_ABS, LOG_RATIO_MAX_ABS);

        const auto log_ratio = aim_log_ratio + rest_log_ratio;
        const auto ratio = torch::exp(log_ratio);

        auto surrogate = torch::zeros_like(log_ratio);
        if (active.aim)
            surrogate = surrogate + group_surrogate(aim_log_ratio, aim_advantages, clip_epsilon);
        if (active.rest)
            surrogate = surrogate + group_surrogate(rest_log_ratio, advantages, clip_epsilon);

        return {
            .surrogate = surrogate,
            .log_ratio = log_ratio,
            .ratio = ratio,
            .kls = {
                .joint = row_kl(log_ratio),
                .aim = row_kl(aim_log_ratio),
                .rest = row_kl(rest_log_ratio)}};
    }

    torch::Tensor continuous_group_mask(
        const ActiveGroups &active, const int64_t nb_continuous_actions,
        const torch::Device device) {
        auto mask = torch::zeros({nb_continuous_actions}, torch::TensorOptions().device(device));
        if (active.rest) mask.slice(0, 0, FIRST_AIM_ACTION).fill_(1.f);
        if (active.aim) mask.slice(0, FIRST_AIM_ACTION).fill_(1.f);
        return mask;
    }

    GroupKlStop::GroupKlStop(const float target_kl)
        : max_kl(target_kl > 0.f ? 1.5f * target_kl : std::numeric_limits<float>::infinity()),
          nb_planned_steps(0), nb_aim_steps(0), nb_rest_steps(0) {}

    const ActiveGroups &GroupKlStop::active() const { return groups; }

    bool GroupKlStop::any_active() const { return groups.aim || groups.rest; }

    bool GroupKlStop::stop_exceeded(const GroupKls &kls) {
        const auto was = groups;
        groups.aim = groups.aim && kls.aim <= max_kl;
        groups.rest = groups.rest && kls.rest <= max_kl;
        return was.aim != groups.aim || was.rest != groups.rest;
    }

    void GroupKlStop::on_planned_steps(const int nb_steps) { nb_planned_steps += nb_steps; }

    void GroupKlStop::on_step() {
        if (groups.aim) nb_aim_steps++;
        if (groups.rest) nb_rest_steps++;
    }

    float GroupKlStop::aim_skip_fraction() const {
        if (nb_planned_steps == 0) return 0.f;
        return 1.f - static_cast<float>(nb_aim_steps) / static_cast<float>(nb_planned_steps);
    }

    float GroupKlStop::rest_skip_fraction() const {
        if (nb_planned_steps == 0) return 0.f;
        return 1.f - static_cast<float>(nb_rest_steps) / static_cast<float>(nb_planned_steps);
    }

}// namespace arenai::agent
