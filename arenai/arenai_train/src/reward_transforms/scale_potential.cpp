//
// Created by samuel on 11/06/2026.
//

#include "./scale_potential.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    ScalePotentialTransform::ScalePotentialTransform(const float potential_reward_scale)
        : potential_reward_scale_(potential_reward_scale) {}

    torch::Tensor ScalePotentialTransform::transform(const torch::Tensor &batch_step_reward) {
        return batch_step_reward * potential_reward_scale_;
    }

    void ScalePotentialTransform::on_add(const torch::Tensor &single_step_reward) {}

}// namespace arenai::train
