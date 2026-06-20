//
// Created by samuel on 10/06/2026.
//

#include "./running_norm_potential.h"

/*
 * Running mean/std
 */
NormalizedPotentialTransform::NormalizedPotentialTransform(
    const int memory_size, const float potential_reward_scale)
    : memory_size_(memory_size), write_idx_(0), size_(0),
      potential_reward_scale_(potential_reward_scale), potential_running_sum_(0.0),
      potential_running_sum_sq_(0.0), potential_reward_history_(memory_size, 0.0) {}

InputRewards NormalizedPotentialTransform::transform(const InputRewards &single_step_rewards) {
    const auto potential_reward_double = single_step_rewards.potential_reward.item<double>();

    if (is_full()) {
        const double old_potential_reward = potential_reward_history_[write_idx_];
        potential_running_sum_ -= old_potential_reward;
        potential_running_sum_sq_ -= old_potential_reward * old_potential_reward;
    }

    potential_reward_history_[write_idx_] = potential_reward_double;
    potential_running_sum_ += potential_reward_double;
    potential_running_sum_sq_ += potential_reward_double * potential_reward_double;

    write_idx_ = (write_idx_ + 1) % memory_size_;
    if (size_ < memory_size_) size_++;

    // transform
    const auto current_size = static_cast<float>(std::max(static_cast<int>(size_), 1));

    const auto potential_reward_mean = static_cast<float>(potential_running_sum_ / current_size);
    const auto potential_reward_var = static_cast<float>(
        potential_running_sum_sq_ / current_size - potential_reward_mean * potential_reward_mean);
    const auto potential_reward_std = std::sqrt(std::max(potential_reward_var, 0.f) + 1e-8f);

    const auto normalized_potential_reward =
        (single_step_rewards.potential_reward - potential_reward_mean) / potential_reward_std;

    return {single_step_rewards.main_reward, potential_reward_scale_ * normalized_potential_reward};
}

bool NormalizedPotentialTransform::is_full() const { return size_ >= memory_size_; }
