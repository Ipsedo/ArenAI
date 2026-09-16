//
// Created by claude on 22/07/2026.
//

#include "./ppo_collector.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    PpoStepCollector::PpoStepCollector(std::shared_ptr<PpoRolloutBuffer> rollout_buffer)
        : rollout_buffer(std::move(rollout_buffer)) {}

    void PpoStepCollector::on_act(
        const TorchState &state, const TorchAction &action,
        const torch::Tensor &continuous_log_prob, const torch::Tensor &discrete_log_prob) {
        last_state = state;
        last_action = action;
        last_continuous_log_prob = continuous_log_prob;
        last_discrete_log_prob = discrete_log_prob;
    }

    void PpoStepCollector::on_transition(
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

    void PpoStepCollector::on_episode_end(const TorchState &final_state) {
        rollout_buffer->finish_episode(final_state);
    }

}// namespace arenai::agent
