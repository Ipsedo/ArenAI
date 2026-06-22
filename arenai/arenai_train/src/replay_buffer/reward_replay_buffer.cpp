//
// Created by samuel on 20/06/2026.
//

#include "./reward_replay_buffer.h"

RewardTransformReplayBuffer::RewardTransformReplayBuffer(
    const int memory_size, const std::shared_ptr<AbstractRewardTransform> &main_reward_transform,
    const std::shared_ptr<AbstractRewardTransform> &potential_reward_transform,
    const std::shared_ptr<AbstractRewardsCombiner> &combiner)
    : ReplayBuffer(memory_size), main_reward_transform_(main_reward_transform),
      potential_reward_transform_(potential_reward_transform), combiner_(combiner) {}

TorchInputStep RewardTransformReplayBuffer::on_add_step(const TorchInputStep &single_step) const {
    return {
        single_step.state,
        single_step.action,
        main_reward_transform_->transform(single_step.main_reward),
        potential_reward_transform_->transform(single_step.potential_reward),
        single_step.done,
        single_step.next_state};
}

TorchOutputStep RewardTransformReplayBuffer::to_output(const TorchInputStep &batch_steps) const {
    return {
        batch_steps.state, batch_steps.action,
        combiner_->to_reward({batch_steps.main_reward, batch_steps.potential_reward}),
        batch_steps.done, batch_steps.next_state};
}
