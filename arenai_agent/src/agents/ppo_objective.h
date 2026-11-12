#ifndef ARENAI_PPO_OBJECTIVE_H
#define ARENAI_PPO_OBJECTIVE_H

#include <torch/torch.h>

namespace arenai::agent {

    // the continuous actions from this index on lay the gun (canon_x, canon_y)
    constexpr int64_t FIRST_AIM_ACTION = 2;

    // [T, nb_tanks, 1] λ-weighted sums of the deltas, each cut stopping the recursion
    torch::Tensor
    generalized_advantages(const torch::Tensor &deltas, const torch::Tensor &cuts, float discount);

    // zero mean and unit variance over the live transitions
    torch::Tensor normalized(const torch::Tensor &advantages, const torch::Tensor &valids);

    // the aim potential's own λ-return: Φ(s_t) is an exact baseline, no critic is needed
    torch::Tensor aim_advantages(
        const torch::Tensor &aim_rewards, const torch::Tensor &cuts, const torch::Tensor &valids,
        float gamma, float lambda);

    // the log-probability of an action split between the gun laying and everything else
    struct PolicyLogProbs {
        // [N, 1]
        torch::Tensor aim;
        torch::Tensor rest;

        // continuous per action [N, nb_continuous_actions], discrete summed [N, 1]
        static PolicyLogProbs
        split(const torch::Tensor &continuous_log_probs, const torch::Tensor &discrete_log_probs);
    };

    struct ClippedSurrogate {
        // [N, 1]
        torch::Tensor surrogate;
        // of the whole action, for the KL and the clip fraction
        torch::Tensor log_ratio;
        torch::Tensor ratio;
    };

    // each action group clipped apart on its own advantage
    ClippedSurrogate clipped_surrogate(
        const PolicyLogProbs &current, const PolicyLogProbs &old,
        const torch::Tensor &aim_advantages, const torch::Tensor &advantages, float clip_epsilon);

}// namespace arenai::agent

#endif//ARENAI_PPO_OBJECTIVE_H
