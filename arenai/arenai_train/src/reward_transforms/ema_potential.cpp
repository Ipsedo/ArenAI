//
// Created by samuel on 10/06/2026.
//

#include "./ema_potential.h"

EmaPotentialTransform::EmaPotentialTransform(
    const float potential_reward_scale, const float ema_decay)
    : potential_reward_ema_decay_(ema_decay), potential_reward_ema_mean_(0.f),
      potential_reward_ema_var_(1.f), ema_initialized_(false),
      potential_reward_scale(potential_reward_scale) {}

void EmaPotentialTransform::on_add(const torch::Tensor &single_step_reward) {
    const auto potential_r = single_step_reward.item<float>();

    if (!ema_initialized_) {
        potential_reward_ema_mean_ = potential_r;
        potential_reward_ema_var_ = 1.f;
        ema_initialized_ = true;

        return;
    }

    const float delta = potential_r - potential_reward_ema_mean_;
    potential_reward_ema_mean_ = potential_reward_ema_decay_ * potential_reward_ema_mean_
                                 + (1.f - potential_reward_ema_decay_) * potential_r;
    potential_reward_ema_var_ = potential_reward_ema_decay_ * potential_reward_ema_var_
                                + (1.f - potential_reward_ema_decay_) * delta * delta;
}

torch::Tensor EmaPotentialTransform::transform(const torch::Tensor &batch_step_reward) {
    const float potential_std = std::sqrt(potential_reward_ema_var_ + 1e-8f);
    const auto norm_potential_reward =
        (batch_step_reward - potential_reward_ema_mean_) / potential_std;

    return potential_reward_scale * norm_potential_reward;
}
