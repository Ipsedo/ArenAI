//
// Created by claude on 27/10/2026.
//

#ifndef ARENAI_LIQUID_IMPALA_PPO_COLLECTOR_H
#define ARENAI_LIQUID_IMPALA_PPO_COLLECTOR_H

#include <memory>

#include "../ppo_liquid/liquid_ppo_rollout_buffer.h"
#include "../step_collector.h"
#include "../torch_types.h"
#include "./liquid_impala_hidden_state.h"

namespace arenai::agent {

    class LiquidImpalaPpoStepCollector final : public AbstractStepCollector {
    public:
        LiquidImpalaPpoStepCollector(
            std::shared_ptr<LiquidPpoRolloutBuffer> rollout_buffer,
            std::shared_ptr<LiquidImpalaHiddenState> hidden_state);

        // concrete act-time channel, called by TorchLiquidImpalaPpoAgent::act
        void on_act(
            const TorchState &state, const TorchAction &action,
            const torch::Tensor &continuous_log_prob, const torch::Tensor &discrete_log_prob,
            const torch::Tensor &actor_hidden);

        void on_transition(
            const torch::Tensor &rewards, const torch::Tensor &done,
            const torch::Tensor &truncated) override;

        void on_episode_end(const TorchState &final_state) override;

    private:
        std::shared_ptr<LiquidPpoRolloutBuffer> rollout_buffer;
        std::shared_ptr<LiquidImpalaHiddenState> hidden_state;

        TorchState last_state;
        TorchAction last_action;
        torch::Tensor last_continuous_log_prob;
        torch::Tensor last_discrete_log_prob;
        torch::Tensor last_actor_hidden;

        // raised at construction and by on_episode_end, consumed by the next on_transition
        bool next_episode_start = true;
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_IMPALA_PPO_COLLECTOR_H
