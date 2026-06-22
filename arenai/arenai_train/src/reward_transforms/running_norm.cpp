//
// Created by samuel on 10/06/2026.
//

#include "./running_norm.h"

/*
 * Running mean/std
 */
NormalizedPotentialTransform::NormalizedPotentialTransform(
    const int memory_size, const float potential_reward_scale)
    : memory_size_(memory_size), write_idx_(0), size_(0),
      potential_reward_scale_(potential_reward_scale), potential_running_sum_(0.0),
      potential_running_sum_sq_(0.0), potential_reward_history_(memory_size, 0.0) {}

torch::Tensor NormalizedPotentialTransform::transform(const torch::Tensor &single_step_rewards) {
    const auto potential_reward_double = single_step_rewards.item<double>();

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
        (single_step_rewards - potential_reward_mean) / potential_reward_std;

    return potential_reward_scale_ * normalized_potential_reward;
}

bool NormalizedPotentialTransform::is_full() const { return size_ >= memory_size_; }

/*
 * Non-zero mean
 */

NormalizedNonZeroTransform::NormalizedNonZeroTransform(const int memory_size)
    : memory_size_(memory_size), write_idx_(0), size_(0), non_zero_nb_(0),
      non_zero_running_sum_sq_(0.0), reward_history_(memory_size, 0.0) {}

torch::Tensor NormalizedNonZeroTransform::transform(const torch::Tensor &single_step_reward) {

    const auto reward_double = single_step_reward.item<double>();

    if (is_full()) {
        const double old_reward = reward_history_[write_idx_];

        non_zero_running_sum_sq_ -= old_reward * old_reward;

        non_zero_nb_ -= old_reward != 0.0 ? 1 : 0;
    }

    reward_history_[write_idx_] = reward_double;

    if (reward_double != 0.0) {
        non_zero_running_sum_sq_ += reward_double * reward_double;

        non_zero_nb_ += 1;
    }

    write_idx_ = (write_idx_ + 1) % memory_size_;
    if (size_ < memory_size_) size_++;

    // Scale-only normalization: keep zeros at zero and preserve the sign of events,
    // only rescale their magnitude by the running RMS of non-zero rewards.
    // Subtracting a mean here would inject a constant reward on every (zero) idle
    // step, which dominates the value function for sparse event rewards.
    const auto current_size = static_cast<float>(std::max(static_cast<int>(non_zero_nb_), 1));
    const auto reward_rms =
        std::sqrt(static_cast<float>(non_zero_running_sum_sq_ / current_size) + 1e-8f);

    return single_step_reward / reward_rms;
}

bool NormalizedNonZeroTransform::is_full() const { return size_ >= memory_size_; }
