#ifndef ARENAI_PPG_COLLECTOR_H
#define ARENAI_PPG_COLLECTOR_H

#include <memory>

#include "../step_collector.h"
#include "../torch_types.h"
#include "./ppg_rollout_buffer.h"

namespace arenai::agent {

    class PpgStepCollector final : public AbstractStepCollector {
    public:
        PpgStepCollector(std::shared_ptr<PpgRolloutBuffer> rollout_buffer);

        // concrete act-time channel, called by TorchPpgAgent::act
        void on_act(
            const TorchState &state, const TorchAction &action,
            const torch::Tensor &continuous_pre_tanh, const torch::Tensor &continuous_log_prob,
            const torch::Tensor &aim_cell, const torch::Tensor &aim_pre_tanh,
            const torch::Tensor &aim_log_prob, const torch::Tensor &discrete_log_prob);

        void on_presence(const torch::Tensor &presence) override;

        void on_transition(
            const torch::Tensor &rewards, const torch::Tensor &done,
            const torch::Tensor &truncated) override;

        void on_episode_end(const TorchState &final_state) override;

    private:
        std::shared_ptr<PpgRolloutBuffer> rollout_buffer;

        TorchState last_state;
        TorchAction last_action;
        torch::Tensor last_continuous_pre_tanh;
        torch::Tensor last_continuous_log_prob;
        torch::Tensor last_aim_cell;
        torch::Tensor last_aim_pre_tanh;
        torch::Tensor last_aim_log_prob;
        torch::Tensor last_discrete_log_prob;
        torch::Tensor last_presence;
    };

}// namespace arenai::agent

#endif//ARENAI_PPG_COLLECTOR_H
