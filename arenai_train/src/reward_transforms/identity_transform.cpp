//
// Created by samuel on 24/06/2026.
//

#include "./identity_transform.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    IdentityTransform::IdentityTransform() = default;

    torch::Tensor IdentityTransform::transform(const torch::Tensor &batch_step_reward) {
        return batch_step_reward;
    }

    void IdentityTransform::on_add(const torch::Tensor &single_step_reward) {}

}// namespace arenai::train
