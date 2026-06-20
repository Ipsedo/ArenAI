//
// Created by samuel on 11/06/2026.
//

#include "./scale_potential.h"

ScalePotentialTransform::ScalePotentialTransform(const float potential_reward_scale)
    : potential_reward_scale_(potential_reward_scale) {}

InputRewards ScalePotentialTransform::transform(const InputRewards &single_step_rewards) {
    return {
        single_step_rewards.main_reward,
        single_step_rewards.potential_reward * potential_reward_scale_};
}
