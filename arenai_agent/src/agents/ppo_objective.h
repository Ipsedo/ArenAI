#ifndef ARENAI_PPO_OBJECTIVE_H
#define ARENAI_PPO_OBJECTIVE_H

#include <vector>

#include <torch/torch.h>

namespace arenai::agent {

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

    struct ClippedSurrogate {
        // [N, 1]
        torch::Tensor surrogate;
        torch::Tensor log_ratio;
        torch::Tensor ratio;
        // [N]
        torch::Tensor kl;
    };

    ClippedSurrogate clipped_surrogate(
        const torch::Tensor &log_probs, const torch::Tensor &old_log_probs,
        const torch::Tensor &advantages, float clip_epsilon);

    // early stopping of the actor epochs
    class KlStop {
    public:
        explicit KlStop(float target_kl);

        bool stopped() const;

        // stops once the KL exceeds 1.5 target_kl; true if stopped
        bool stop_exceeded(float kl);

        void on_planned_steps(int nb_steps);
        void on_step();

        float skip_fraction() const;

    private:
        float max_kl;
        bool is_stopped;
        int nb_planned_steps;
        int nb_steps;
    };

}// namespace arenai::agent

#endif//ARENAI_PPO_OBJECTIVE_H
