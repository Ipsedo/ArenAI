//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_STEP_COLLECTOR_H
#define ARENAI_STEP_COLLECTOR_H

#include <torch/torch.h>

#include "./torch_types.h"

namespace arenai::train {

    // Env-feedback ingestion. The act-time recording (state, action and any
    // algorithm-specific extras) goes through the concrete agent -> collector
    // channel, wired by the factory.
    class AbstractStepCollector {
    public:
        virtual ~AbstractStepCollector() = default;

        virtual void on_transition(
            const torch::Tensor &rewards, const torch::Tensor &done,
            const torch::Tensor &truncated) = 0;

        virtual void on_episode_end(const TorchState &final_state) = 0;
    };

}// namespace arenai::train

#endif//ARENAI_STEP_COLLECTOR_H
