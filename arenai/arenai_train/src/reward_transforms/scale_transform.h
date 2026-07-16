//
// Created by samuel on 11/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_SCALE_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_SCALE_REPLAY_BUFFER_H

#include "./reward_transform.h"

namespace arenai::train {

    class ScalePotentialTransform : public AbstractRewardTransform {
    public:
        explicit ScalePotentialTransform(float potential_reward_scale);

        void on_add(const torch::Tensor &single_step_reward) override;

        torch::Tensor transform(const torch::Tensor &batch_step_reward) override;

    private:
        float potential_reward_scale_;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_SCALE_REPLAY_BUFFER_H
