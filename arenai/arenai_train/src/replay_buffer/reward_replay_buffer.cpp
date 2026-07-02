//
// Created by samuel on 20/06/2026.
//

#include "./reward_replay_buffer.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    RewardTransformReplayBuffer::RewardTransformReplayBuffer(
        const int memory_size, const std::shared_ptr<AbstractRewardTransform> &reward_transform)
        : ReplayBuffer(memory_size), reward_transform_(reward_transform) {}

    void RewardTransformReplayBuffer::on_add_step(const TorchStep &single_step) const {
        reward_transform_->on_add(single_step.reward);
    }

    TorchStep RewardTransformReplayBuffer::transform_at_sample(const TorchStep &batch_steps) const {
        return {
            batch_steps.state, batch_steps.action, reward_transform_->transform(batch_steps.reward),
            batch_steps.done, batch_steps.next_state};
    }

}// namespace arenai::train
