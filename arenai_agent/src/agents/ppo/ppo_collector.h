//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_PPO_COLLECTOR_H
#define ARENAI_PPO_COLLECTOR_H

#include <memory>

#include "../step_collector.h"
#include "../torch_types.h"
#include "./ppo_rollout_buffer.h"

namespace arenai::agent {

    class PpoStepCollector final : public AbstractStepCollector {
    public:
        explicit PpoStepCollector(std::shared_ptr<PpoRolloutBuffer> rollout_buffer);

        // concrete act-time channel, called by TorchPpoAgent::act
        void on_act(
            const TorchState &state, const TorchAction &action,
            const torch::Tensor &continuous_log_prob, const torch::Tensor &discrete_log_prob);

        void on_transition(
            const torch::Tensor &rewards, const torch::Tensor &done,
            const torch::Tensor &truncated) override;

        void on_episode_end(const TorchState &final_state) override;

    private:
        std::shared_ptr<PpoRolloutBuffer> rollout_buffer;

        TorchState last_state;
        TorchAction last_action;
        torch::Tensor last_continuous_log_prob;
        torch::Tensor last_discrete_log_prob;
    };

}// namespace arenai::agent

#endif//ARENAI_PPO_COLLECTOR_H
