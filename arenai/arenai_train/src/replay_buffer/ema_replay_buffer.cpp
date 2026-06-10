//
// Created by samuel on 10/06/2026.
//

#include "./ema_replay_buffer.h"

PotentialRewardEmaReplayBuffer::PotentialRewardEmaReplayBuffer(
    const int memory_size, const float potential_reward_scale, const float ema_decay)
    : ReplayBuffer(memory_size), potential_reward_ema_decay_(ema_decay),
      potential_reward_ema_mean_(0.f), potential_reward_ema_var_(1.f), ema_initialized_(false),
      potential_reward_scale(potential_reward_scale) {}

void PotentialRewardEmaReplayBuffer::on_add_step(const int write_idx, const TorchInputStep &step) {
    const auto potential_r = step.potential_reward.item<float>();

    if (!ema_initialized_) {
        potential_reward_ema_mean_ = potential_r;
        potential_reward_ema_var_ = 1.f;
        ema_initialized_ = true;
    } else {
        const float delta = potential_r - potential_reward_ema_mean_;
        potential_reward_ema_mean_ = potential_reward_ema_decay_ * potential_reward_ema_mean_
                                     + (1.f - potential_reward_ema_decay_) * potential_r;
        potential_reward_ema_var_ = potential_reward_ema_decay_ * potential_reward_ema_var_
                                    + (1.f - potential_reward_ema_decay_) * delta * delta;
    }
}

TorchOutputStep PotentialRewardEmaReplayBuffer::to_output_step(const TorchInputStep &batch_steps) {
    const float potential_std = std::sqrt(potential_reward_ema_var_ + 1e-8f);
    const auto norm_potential_reward =
        (batch_steps.potential_reward - potential_reward_ema_mean_) / potential_std;

    return {
        batch_steps.state, batch_steps.action,
        potential_reward_scale * norm_potential_reward + batch_steps.main_reward, batch_steps.done,
        batch_steps.next_state};
}
