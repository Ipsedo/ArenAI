#ifndef ARENAI_STEP_COLLECTOR_H
#define ARENAI_STEP_COLLECTOR_H

#include "./torch_types.h"

namespace arenai::agent {

    class AbstractStepCollector {
    public:
        virtual ~AbstractStepCollector() = default;

        // aim_rewards: the aim potential part of the rewards, which the gun laying learns from
        virtual void on_transition(
            const torch::Tensor &rewards, const torch::Tensor &aim_rewards,
            const torch::Tensor &done, const torch::Tensor &truncated) = 0;

        virtual void on_episode_end(const TorchState &final_state) = 0;
    };

}// namespace arenai::agent

#endif//ARENAI_STEP_COLLECTOR_H
