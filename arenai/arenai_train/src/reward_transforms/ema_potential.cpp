//
// Created by samuel on 10/06/2026.
//

#include "./ema_potential.h"

EmaPotentialTransform::EmaPotentialTransform(
    const float potential_reward_scale, const float ema_decay)
    : potential_reward_ema_decay_(ema_decay), potential_reward_ema_mean_(0.f),
      potential_reward_ema_var_(1.f), ema_initialized_(false),
      potential_reward_scale(potential_reward_scale) {}

InputRewards EmaPotentialTransform::transform(const InputRewards &single_step_rewards) {
    const auto potential_r = single_step_rewards.potential_reward.item<float>();

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

    const float potential_std = std::sqrt(potential_reward_ema_var_ + 1e-8f);
    const auto norm_potential_reward =
        (single_step_rewards.potential_reward - potential_reward_ema_mean_) / potential_std;

    return {single_step_rewards.main_reward, potential_reward_scale * norm_potential_reward};
}
