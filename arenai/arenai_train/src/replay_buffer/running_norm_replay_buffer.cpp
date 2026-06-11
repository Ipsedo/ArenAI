//
// Created by samuel on 10/06/2026.
//

#include "./running_norm_replay_buffer.h"

/*
 * Running mean/std
 */
NormalizedPotentialRewardReplayBuffer::NormalizedPotentialRewardReplayBuffer(
    const int memory_size, const float potential_reward_scale)
    : ReplayBuffer(memory_size), potential_reward_scale_(potential_reward_scale),
      potential_running_sum_(0.0), potential_running_sum_sq_(0.0),
      potential_reward_history_(memory_size, 0.0) {}

TorchInputStep NormalizedPotentialRewardReplayBuffer::on_add_step(
    const int write_idx, const TorchInputStep &step) {
    const auto potential_reward_double = step.potential_reward.item<double>();

    if (is_full()) {
        const double old_potential_reward = potential_reward_history_[write_idx];
        potential_running_sum_ -= old_potential_reward;
        potential_running_sum_sq_ -= old_potential_reward * old_potential_reward;
    }

    potential_reward_history_[write_idx] = potential_reward_double;
    potential_running_sum_ += potential_reward_double;
    potential_running_sum_sq_ += potential_reward_double * potential_reward_double;

    return step;
}

TorchOutputStep
NormalizedPotentialRewardReplayBuffer::to_output_step(const TorchInputStep &batch_steps) {
    const auto current_size = static_cast<float>(std::max(size(), 1));

    const auto potential_reward_mean = static_cast<float>(potential_running_sum_ / current_size);
    const auto potential_reward_var = static_cast<float>(
        potential_running_sum_sq_ / current_size - potential_reward_mean * potential_reward_mean);
    const auto potential_reward_std = std::sqrt(std::max(potential_reward_var, 0.f) + 1e-8f);

    const auto normalized_potential_reward =
        (batch_steps.potential_reward - potential_reward_mean) / potential_reward_std;

    return {
        batch_steps.state, batch_steps.action,
        potential_reward_scale_ * normalized_potential_reward + batch_steps.main_reward,
        batch_steps.done, batch_steps.next_state};
}
