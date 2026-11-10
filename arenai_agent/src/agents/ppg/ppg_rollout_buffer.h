#ifndef ARENAI_PPG_ROLLOUT_BUFFER_H
#define ARENAI_PPG_ROLLOUT_BUFFER_H

#include <optional>
#include <vector>

#include <torch/torch.h>

#include "../torch_types.h"

namespace arenai::agent {

    struct PpgInputStep {
        TorchState state;
        TorchAction action;
        torch::Tensor continuous_pre_tanh;
        torch::Tensor continuous_log_prob;
        torch::Tensor discrete_log_prob;
        torch::Tensor reward;
        torch::Tensor done;
        torch::Tensor truncated;
    };

    struct PpgRollout {
        TorchState states;
        TorchAction actions;
        torch::Tensor continuous_pre_tanhs;
        torch::Tensor continuous_log_probs;
        torch::Tensor discrete_log_probs;
        torch::Tensor rewards;
        torch::Tensor dones;
        torch::Tensor truncateds;
        // [nb_tanks, ...]
        TorchState bootstrap_state;
        // [T, nb_tanks, 1]
        torch::Tensor valids;
    };

    class PpgRolloutBuffer {
    public:
        void add(const PpgInputStep &step);
        void finish_episode(const TorchState &final_state);

        size_t nb_complete_steps() const;

        PpgRollout get_rollout();

    private:
        struct StoredStep {
            PpgInputStep step;
            torch::Tensor valid;
        };

        std::vector<StoredStep> steps_;

        std::optional<TorchState> final_state_;

        // [nb_tanks]
        torch::Tensor already_terminated_;
    };

}// namespace arenai::agent

#endif//ARENAI_PPG_ROLLOUT_BUFFER_H
