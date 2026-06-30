//
// Created by samuel on 12/06/2026.
//

#include "./delta_scale_potential.h"

#include <arenai_core/constants.h>
#include <arenai_model/constants.h>

DeltaScalePotentialRewardTransform::DeltaScalePotentialRewardTransform(
    const float wanted_frequency, const float target_potential_reward)
    : ScalePotentialTransform(compute_potential_reward_scale(
        WHEEL_RADIAL_VELOCITY, 1.1f, ENEMY_TURRET_RADIAL_VELOCITY, ENEMY_TURRET_RADIAL_VELOCITY,
        250.f, wanted_frequency, target_potential_reward)) {}

float DeltaScalePotentialRewardTransform::compute_potential_reward_scale(
    const float wheel_radial_velocity, const float wheel_radius,
    const float turret_angular_velocity, const float canon_angular_velocity,
    const float distance_scale, const float dt, const float target_mean_potential_reward) {

    const float linear_velocity = wheel_radial_velocity * wheel_radius;

    const float max_distance_gradient = 1.f / (distance_scale * std::sqrt(std::exp(1.f)));

    constexpr float max_angle_gradient = 0.5f;

    const float max_delta_phi_distance = max_distance_gradient * linear_velocity * dt;
    const float max_delta_phi_angle =
        max_angle_gradient * (turret_angular_velocity + canon_angular_velocity) * dt;

    const float max_delta_phi = max_delta_phi_distance + max_delta_phi_angle;

    constexpr float typical_fraction = 0.1f;
    const float typical_delta_phi = typical_fraction * max_delta_phi;

    return target_mean_potential_reward / typical_delta_phi;
}
