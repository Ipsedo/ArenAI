//
// Created by samuel on 12/06/2026.
//

#include "./delta_scale_potential.h"

#include <arenai_core/constants.h>
#include <arenai_model/constants.h>

DeltaScalePotentialRewardTransform::DeltaScalePotentialRewardTransform(
    const float wanted_frequency, const float target_potential_reward)
    : potential_reward_scale_(
        compute_potential_reward_scale(wanted_frequency, 500.f, target_potential_reward, 0.1f)) {}

float DeltaScalePotentialRewardTransform::compute_potential_reward_scale(
    const float wanted_frequency, const float distance_scale, const float target_reward,
    const float typical_fraction) {
    const float max_grad_distance = std::sqrt(2.f / std::exp(1.f)) / distance_scale;

    const float max_delta_distance_score =
        max_grad_distance * WHEEL_RADIAL_VELOCITY * wanted_frequency;
    const float max_delta_angle_score = 0.5f * ENEMY_TURRET_RADIAL_VELOCITY * wanted_frequency;

    const float max_delta_phi = max_delta_distance_score + max_delta_angle_score;
    const float typical_delta_phi = typical_fraction * max_delta_phi;

    return target_reward / typical_delta_phi;
}

InputRewards
DeltaScalePotentialRewardTransform::transform(const InputRewards &single_step_rewards) {
    return {
        single_step_rewards.main_reward,
        potential_reward_scale_ * single_step_rewards.potential_reward};
}
