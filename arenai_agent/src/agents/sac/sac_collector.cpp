//
// Created by claude on 22/07/2026.
//

#include "./sac_collector.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    SacStepCollector::SacStepCollector(std::shared_ptr<SacReplayBuffer> replay_buffer)
        : replay_buffer(std::move(replay_buffer)) {}

    void SacStepCollector::on_act(const TorchState &state, const TorchAction &action) {
        last_state = state;
        last_action = action;
    }

    void SacStepCollector::on_transition(
        const torch::Tensor &rewards, const torch::Tensor &done, const torch::Tensor &truncated) {
        replay_buffer->add(
            {.state = last_state,
             .action = last_action,
             .reward = rewards,
             .done = done,
             .truncated = truncated});
    }

    void SacStepCollector::on_episode_end(const TorchState &final_state) {
        replay_buffer->finish_episode(final_state);
    }

}// namespace arenai::agent
