#ifndef ARENAI_PPO_ROLLOUT_BUFFER_H
#define ARENAI_PPO_ROLLOUT_BUFFER_H

#include <optional>
#include <vector>

#include <torch/torch.h>

#include "../torch_types.h"

namespace arenai::agent {

    struct PpoInputStep {
        TorchState state;
        TorchAction action;
        // pre-tanh draw of the squashed normal, what the log-probabilities are computed on
        torch::Tensor continuous_pre_tanh;
        torch::Tensor continuous_log_prob;
        torch::Tensor discrete_log_prob;
        torch::Tensor reward;
        torch::Tensor done;
        // segment end that is not a death: cuts the return like a done, the tank plays on
        torch::Tensor truncated;
        // [nb_tanks, rows, cols] enemy presence map of the state
        torch::Tensor presence;
    };

    // On-policy rollout stacked on the time dimension: every tensor is [T, nb_tanks, ...]
    struct PpoRollout {
        TorchState states;
        TorchAction actions;
        torch::Tensor continuous_pre_tanhs;
        torch::Tensor continuous_log_probs;
        torch::Tensor discrete_log_probs;
        torch::Tensor rewards;
        torch::Tensor dones;
        // segment ends that are not deaths: cut the return like dones, the tanks play on
        torch::Tensor truncateds;
        torch::Tensor presences;
        // [nb_tanks, ...] observation closing the last step, for the value bootstrap
        TorchState bootstrap_state;
        // [T, nb_tanks, 1] whether the (step, tank) pair is a live transition
        torch::Tensor valids;
        // [T] bool, the step closes an episode
        torch::Tensor episode_ends;
        // one per episode end, in step order
        std::vector<TorchState> closing_states;
    };

    class PpoRolloutBuffer {
    public:
        void add(const PpoInputStep &step);
        void finish_episode(const TorchState &final_state);

        size_t nb_complete_steps() const;

        // stacks and removes every complete step; the pending one (if any) stays
        PpoRollout get_rollout();

    private:
        struct StoredStep {
            PpoInputStep step;
            torch::Tensor valid;
            std::optional<TorchState> closing_state;
            bool episode_start;
        };

        std::vector<StoredStep> steps_;

        // observation closing the last stored step, set by finish_episode()
        std::optional<TorchState> final_state_;

        // [nb_tanks] tanks already done in the current episode
        torch::Tensor already_terminated_;

        bool next_step_starts_episode_ = true;
    };

}// namespace arenai::agent

#endif//ARENAI_PPO_ROLLOUT_BUFFER_H
