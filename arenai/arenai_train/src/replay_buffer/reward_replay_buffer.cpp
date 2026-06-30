//
// Created by samuel on 20/06/2026.
//

#include "./reward_replay_buffer.h"

RewardTransformReplayBuffer::RewardTransformReplayBuffer(
    const int memory_size, const std::shared_ptr<AbstractRewardTransform> &reward_transform,
    const std::shared_ptr<AbstractRewardTransform> &potential_transform)
    : ReplayBuffer(memory_size), reward_transform_(reward_transform),
      potential_transform_(potential_transform) {}

void RewardTransformReplayBuffer::on_add_step(const TorchInputStep &single_step) const {
    reward_transform_->on_add(single_step.reward);
    potential_transform_->on_add(single_step.potential);
}

TorchOutputStep
RewardTransformReplayBuffer::transform_at_sample(const TorchInputStep &batch_steps) const {
    return {
        batch_steps.state, batch_steps.action,
        reward_transform_->transform(batch_steps.reward)
            + potential_transform_->transform(batch_steps.potential),
        batch_steps.done, batch_steps.next_state};
}
