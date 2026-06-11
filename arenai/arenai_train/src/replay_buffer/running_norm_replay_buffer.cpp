//
// Created by samuel on 10/06/2026.
//

#include "./running_norm_replay_buffer.h"

/*
 * Running mean/std
 */
NormalizedPotentialRewardReplayBuffer::NormalizedPotentialRewardReplayBuffer(
    const int memory_size, const float potential_reward_scale)
    : ReplayBuffer(memory_size), potential_reward_scale_(potential_reward_scale), running_sum_(0.0),
      running_sum_sq_(0.0), reward_history_(memory_size, 0.f) {}

TorchInputStep NormalizedPotentialRewardReplayBuffer::on_add_step(
    const int write_idx, const TorchInputStep &step) {
    const auto r = step.potential_reward.item<double>();

    if (is_full()) {
        const double old_r = reward_history_[write_idx];
        running_sum_ -= old_r;
        running_sum_sq_ -= old_r * old_r;
    }

    reward_history_[write_idx] = r;
    running_sum_ += r;
    running_sum_sq_ += r * r;

    const float current_size = size() > 0 ? static_cast<float>(size()) : 1.f;

    const auto [state, action, main_reward, potential_reward, done, next_state] = step;

    const auto mean = static_cast<float>(running_sum_ / current_size);
    const auto var = static_cast<float>(running_sum_sq_ / current_size - mean * mean);
    const auto std = std::sqrt(std::max(var, 0.f) + 1e-8f);

    const torch::Tensor normalized_potential_reward = (potential_reward - mean) / std;

    return {state, action, main_reward, normalized_potential_reward, done, next_state};
}

TorchOutputStep
NormalizedPotentialRewardReplayBuffer::to_output_step(const TorchInputStep &batch_steps) {
    return {
        batch_steps.state, batch_steps.action,
        potential_reward_scale_ * batch_steps.potential_reward + batch_steps.main_reward,
        batch_steps.done, batch_steps.next_state};
}
