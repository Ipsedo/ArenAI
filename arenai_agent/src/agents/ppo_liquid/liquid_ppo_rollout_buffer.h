//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_LIQUID_PPO_ROLLOUT_BUFFER_H
#define ARENAI_LIQUID_PPO_ROLLOUT_BUFFER_H

#include <optional>
#include <vector>

#include <torch/torch.h>

#include "../torch_types.h"

namespace arenai::agent {

    struct LiquidPpoInputStep {
        TorchState state;
        TorchAction action;
        torch::Tensor continuous_log_prob;
        torch::Tensor discrete_log_prob;
        torch::Tensor reward;
        torch::Tensor done;
        // [nb_tanks, neuron_number] actor liquid state opening the step
        torch::Tensor actor_hidden;
        // the step opens a fresh episode (the liquid states were re-drawn)
        bool episode_start;
    };

    // On-policy rollout stacked on the time dimension: every tensor is [T, nb_tanks, ...]
    struct LiquidPpoRollout {
        TorchState states;
        TorchAction actions;
        torch::Tensor continuous_log_probs;
        torch::Tensor discrete_log_probs;
        torch::Tensor rewards;
        torch::Tensor dones;
        // [nb_tanks, ...] observation closing the last step, for the value bootstrap
        TorchState bootstrap_state;
        // [T, nb_tanks, 1] whether the (step, tank) pair is a live transition
        torch::Tensor valids;
        // [T, nb_tanks, neuron_number] actor liquid state opening each step
        torch::Tensor actor_hiddens;
        // [T] bool, the step opens a fresh episode
        torch::Tensor episode_starts;
    };

    // Sequential on-policy buffer. A step is "complete" once the observation that
    // follows it is known - the next add()'s state, or finish_episode()'s final state.
    class LiquidPpoRolloutBuffer {
    public:
        void add(const LiquidPpoInputStep &step);
        void finish_episode(const TorchState &final_state);

        size_t nb_complete_steps() const;

        // stacks and removes every complete step; the pending one (if any) stays
        LiquidPpoRollout get_rollout();

    private:
        struct StoredStep {
            LiquidPpoInputStep step;
            torch::Tensor valid;
        };

        std::vector<StoredStep> steps_;

        // observation closing the last stored step, set by finish_episode()
        std::optional<TorchState> final_state_;

        // [nb_tanks] tanks already done in the current episode
        torch::Tensor already_terminated_;
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_PPO_ROLLOUT_BUFFER_H
