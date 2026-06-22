//
// Created by samuel on 11/06/2026.
//

#include "./scale_potential.h"

ScalePotentialTransform::ScalePotentialTransform(const float potential_reward_scale)
    : potential_reward_scale_(potential_reward_scale) {}

torch::Tensor ScalePotentialTransform::transform(const torch::Tensor &single_step_reward) {
    return single_step_reward * potential_reward_scale_;
}
