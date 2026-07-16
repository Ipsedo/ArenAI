//
// Created by samuel on 24/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_IDENTITY_TRANSFORM_H
#define ARENAI_TRAIN_HOST_IDENTITY_TRANSFORM_H

#include "./reward_transform.h"

namespace arenai::train {

    class IdentityTransform : public AbstractRewardTransform {
    public:
        IdentityTransform();

        torch::Tensor transform(const torch::Tensor &batch_step_reward) override;
        void on_add(const torch::Tensor &single_step_reward) override;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_IDENTITY_TRANSFORM_H
