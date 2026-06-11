//
// Created by samuel on 11/06/2026.
//

#include "./scale_replay_buffer.h"

ScalePotentialRewardReplayBuffer::ScalePotentialRewardReplayBuffer(
    int memory_size, float potential_reward_scale)
    : ReplayBuffer(memory_size), potential_reward_scale_(potential_reward_scale) {}

TorchInputStep
ScalePotentialRewardReplayBuffer::on_add_step(int write_idx, const TorchInputStep &step) {
    return step;
}

TorchOutputStep
ScalePotentialRewardReplayBuffer::to_output_step(const TorchInputStep &batch_steps) {
    return {
        batch_steps.state, batch_steps.action,
        potential_reward_scale_ * batch_steps.potential_reward + batch_steps.main_reward,
        batch_steps.done, batch_steps.next_state};
}
