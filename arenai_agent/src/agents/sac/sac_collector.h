//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_SAC_COLLECTOR_H
#define ARENAI_SAC_COLLECTOR_H

#include <memory>

#include "../step_collector.h"
#include "../torch_types.h"
#include "./replay_buffer.h"

namespace arenai::agent {

    class SacStepCollector final : public AbstractStepCollector {
    public:
        explicit SacStepCollector(std::shared_ptr<SacReplayBuffer> replay_buffer);

        // concrete act-time channel, called by TrainableSacAgent::act
        void on_act(const TorchState &state, const TorchAction &action);

        void on_transition(
            const torch::Tensor &rewards, const torch::Tensor &done,
            const torch::Tensor &truncated) override;

        void on_episode_end(const TorchState &final_state) override;

    private:
        std::shared_ptr<SacReplayBuffer> replay_buffer;

        TorchState last_state;
        TorchAction last_action;
    };

}// namespace arenai::agent

#endif//ARENAI_SAC_COLLECTOR_H
