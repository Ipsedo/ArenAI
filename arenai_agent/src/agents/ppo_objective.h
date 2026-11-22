#ifndef ARENAI_PPO_OBJECTIVE_H
#define ARENAI_PPO_OBJECTIVE_H

#include <vector>

#include <torch/torch.h>

namespace arenai::agent {

    // the continuous actions from this index on lay the gun (canon_x, canon_y)
    constexpr int64_t FIRST_AIM_ACTION = 2;

    // [T, nb_tanks, 1] λ-weighted sums of the deltas, each cut stopping the recursion
    torch::Tensor
    generalized_advantages(const torch::Tensor &deltas, const torch::Tensor &cuts, float discount);

    // [T, nb_tanks, 1] 1 where the episode ends or is truncated
    torch::Tensor episode_cuts(const torch::Tensor &dones, const torch::Tensor &truncateds);

    // cuts [T, nb_tanks, 1] raised on every tank where episode_ends [T] is set
    torch::Tensor with_episode_ends(const torch::Tensor &cuts, const torch::Tensor &episode_ends);

    // next_values [T, nb_tanks, 1] with the closing values at the episode ends
    torch::Tensor closed_next_values(
        const torch::Tensor &next_values, const torch::Tensor &episode_ends,
        const std::vector<torch::Tensor> &closing_values);

    // zero mean and unit variance over the live transitions
    torch::Tensor normalized(const torch::Tensor &advantages, const torch::Tensor &valids);

    // the log-probability of an action split between the gun laying and everything else
    struct PolicyLogProbs {
        // [N, 1]
        torch::Tensor aim;
        torch::Tensor rest;

        // continuous per action [N, nb_continuous_actions], discrete summed [N, 1]
        static PolicyLogProbs
        split(const torch::Tensor &continuous_log_probs, const torch::Tensor &discrete_log_probs);
    };

    // the action groups an update still trains
    struct ActiveGroups {
        bool aim = true;
        bool rest = true;
    };

    // per-row KL estimates, [N]
    struct RowKls {
        torch::Tensor joint;
        torch::Tensor aim;
        torch::Tensor rest;
    };

    struct GroupKls {
        float joint;
        float aim;
        float rest;
    };

    GroupKls trimmed_group_kls(const std::vector<RowKls> &row_kls);

    struct GroupClippedSurrogate {
        // [N, 1] of the active groups
        torch::Tensor surrogate;
        // of the whole action
        torch::Tensor log_ratio;
        torch::Tensor ratio;
        RowKls kls;
    };

    // each action group clipped apart on its own advantage
    GroupClippedSurrogate clipped_surrogate(
        const PolicyLogProbs &current, const PolicyLogProbs &old,
        const torch::Tensor &aim_advantages, const torch::Tensor &advantages, float clip_epsilon,
        const ActiveGroups &active = {});

    // [nb_continuous_actions] 1 for the actions of an active group
    torch::Tensor continuous_group_mask(
        const ActiveGroups &active, int64_t nb_continuous_actions, torch::Device device);

    // early stopping of the actor epochs, each group on its own KL
    class GroupKlStop {
    public:
        explicit GroupKlStop(float target_kl);

        const ActiveGroups &active() const;
        bool any_active() const;

        // stops the groups whose KL exceeds 1.5 target_kl; true if one was stopped
        bool stop_exceeded(const GroupKls &kls);

        void on_planned_steps(int nb_steps);
        void on_step();

        float aim_skip_fraction() const;
        float rest_skip_fraction() const;

    private:
        float max_kl;
        ActiveGroups groups;
        int nb_planned_steps;
        int nb_aim_steps;
        int nb_rest_steps;
    };

}// namespace arenai::agent

#endif//ARENAI_PPO_OBJECTIVE_H
