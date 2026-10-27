//
// Created by claude on 27/10/2026.
//

#include "./liquid_impala_ppo_collector.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    LiquidImpalaPpoStepCollector::LiquidImpalaPpoStepCollector(
        std::shared_ptr<LiquidPpoRolloutBuffer> rollout_buffer,
        std::shared_ptr<LiquidImpalaHiddenState> hidden_state)
        : rollout_buffer(std::move(rollout_buffer)), hidden_state(std::move(hidden_state)) {}

    void LiquidImpalaPpoStepCollector::on_act(
        const TorchState &state, const TorchAction &action,
        const torch::Tensor &continuous_log_prob, const torch::Tensor &discrete_log_prob,
        const torch::Tensor &actor_hidden) {
        last_state = state;
        last_action = action;
        last_continuous_log_prob = continuous_log_prob;
        last_discrete_log_prob = discrete_log_prob;
        last_actor_hidden = actor_hidden;
    }

    void LiquidImpalaPpoStepCollector::on_transition(
        const torch::Tensor &rewards, const torch::Tensor &done, const torch::Tensor &truncated) {
        rollout_buffer->add(
            {.state = last_state,
             .action = last_action,
             .continuous_log_prob = last_continuous_log_prob,
             .discrete_log_prob = last_discrete_log_prob,
             .reward = rewards,
             .done = done,
             .truncated = truncated,
             .actor_hidden = last_actor_hidden,
             .episode_start = next_episode_start});

        next_episode_start = false;
    }

    void LiquidImpalaPpoStepCollector::on_episode_end(const TorchState &final_state) {
        rollout_buffer->finish_episode(final_state);

        // the environment resets every tank: re-draw the liquid states
        hidden_state->reset();
        next_episode_start = true;
    }

}// namespace arenai::agent
