//
// Created by claude on 22/07/2026.
//

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
        torch::Tensor continuous_log_prob;
        torch::Tensor discrete_log_prob;
        torch::Tensor reward;
        torch::Tensor done;
        torch::Tensor truncated;
    };

    // On-policy rollout stacked on the time dimension: every tensor is [T, nb_tanks, ...]
    struct PpoRollout {
        TorchState states;
        TorchAction actions;
        torch::Tensor continuous_log_probs;
        torch::Tensor discrete_log_probs;
        torch::Tensor rewards;
        torch::Tensor dones;
        torch::Tensor truncateds;
        TorchState next_states;
        // [T, nb_tanks, 1] whether the (step, tank) pair is a live transition
        torch::Tensor valids;
    };

    // Sequential on-policy buffer. A step is "complete" once its next state is
    // known - filled by the following add() or by finish_episode().
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
            std::optional<TorchState> next_state;
        };

        std::vector<StoredStep> steps_;

        // [nb_tanks] tanks already done/truncated in the current episode
        torch::Tensor already_terminated_;
    };

}// namespace arenai::agent

#endif//ARENAI_PPO_ROLLOUT_BUFFER_H
