//
// Created by samuel on 22/07/2026.
//

#include "./ppg_collector.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    PpgStepCollector::PpgStepCollector(std::shared_ptr<PpgRolloutBuffer> rollout_buffer)
        : rollout_buffer(std::move(rollout_buffer)) {}

    void PpgStepCollector::on_act(
        const TorchState &state, const TorchAction &action,
        const torch::Tensor &continuous_log_prob, const torch::Tensor &discrete_log_prob) {
        last_state = state;
        last_action = action;
        last_continuous_log_prob = continuous_log_prob;
        last_discrete_log_prob = discrete_log_prob;
    }

    void PpgStepCollector::on_transition(
        const torch::Tensor &rewards, const torch::Tensor &done, const torch::Tensor &truncated) {
        rollout_buffer->add(
            {.state = last_state,
             .action = last_action,
             .continuous_log_prob = last_continuous_log_prob,
             .discrete_log_prob = last_discrete_log_prob,
             .reward = rewards,
             .done = done,
             .truncated = truncated});
    }

    void PpgStepCollector::on_episode_end(const TorchState &final_state) {
        rollout_buffer->finish_episode(final_state);
    }

}// namespace arenai::agent
