//
// Created by samuel on 22/07/2026.
//

#ifndef ARENAI_EMPTY_STEP_COLLECTOR_H
#define ARENAI_EMPTY_STEP_COLLECTOR_H

#include <torch/torch.h>

#include "./step_collector.h"

namespace arenai::agent {
    class EmptyStepCollector final : public AbstractStepCollector {
    public:
        explicit EmptyStepCollector();

        void on_episode_end(const TorchState &final_state) override;
        void on_transition(
            const torch::Tensor &rewards, const torch::Tensor &done,
            const torch::Tensor &truncated) override;
    };
}// namespace arenai::agent

#endif//ARENAI_EMPTY_STEP_COLLECTOR_H
