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
      potential_running_sum_(0.0), potential_running_sum_sq_(0.0), main_running_sum_(0.0),
      main_running_sum_sq_(0.0), potential_reward_history_(memory_size, 0.0),
      main_reward_history_(memory_size, 0.0) {}

TorchInputStep NormalizedPotentialRewardReplayBuffer::on_add_step(
    const int write_idx, const TorchInputStep &step) {
    const auto potential_reward_double = step.potential_reward.item<double>();
    const auto main_reward_double = step.main_reward.item<double>();

    if (is_full()) {
        const double old_potential_reward = potential_reward_history_[write_idx];
        potential_running_sum_ -= old_potential_reward;
        potential_running_sum_sq_ -= old_potential_reward * old_potential_reward;

        const double old_main_reward = main_reward_history_[write_idx];
        main_running_sum_ -= old_main_reward;
        main_running_sum_sq_ -= old_main_reward * old_main_reward;
    }

    potential_reward_history_[write_idx] = potential_reward_double;
    potential_running_sum_ += potential_reward_double;
    potential_running_sum_sq_ += potential_reward_double * potential_reward_double;

    main_reward_history_[write_idx] = main_running_sum_;
    main_running_sum_ += main_reward_double;
    main_running_sum_sq_ += main_reward_double * main_reward_double;

    const float current_size = size() > 0 ? static_cast<float>(size()) : 1.f;

    const auto [state, action, main_reward, potential_reward, done, next_state] = step;

    const auto potential_mean = static_cast<float>(potential_running_sum_ / current_size);
    const auto potential_var = static_cast<float>(
        potential_running_sum_sq_ / current_size - potential_mean * potential_mean);
    const auto potential_std = std::sqrt(std::max(potential_var, 0.f) + 1e-8f);

    const torch::Tensor normalized_potential_reward =
        (potential_reward - potential_mean) / potential_std;

    const auto main_mean = static_cast<float>(main_running_sum_ / current_size);
    const auto main_var =
        static_cast<float>(main_running_sum_sq_ / current_size - main_mean * main_mean);
    const auto main_std = std::sqrt(std::max(main_var, 0.f) + 1e-8f);

    const torch::Tensor normalized_main_reward = (main_reward - main_mean) / main_std;

    return {state, action, normalized_main_reward, normalized_potential_reward, done, next_state};
}

TorchOutputStep
NormalizedPotentialRewardReplayBuffer::to_output_step(const TorchInputStep &batch_steps) {
    return {
        batch_steps.state, batch_steps.action,
        potential_reward_scale_ * batch_steps.potential_reward + batch_steps.main_reward,
        batch_steps.done, batch_steps.next_state};
}
