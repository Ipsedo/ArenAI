//
// Created by samuel on 10/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_EMA_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_EMA_REPLAY_BUFFER_H

#include "./reward_transform.h"

namespace arenai::train {

    class EmaPotentialTransform : public AbstractRewardTransform {
    public:
        explicit EmaPotentialTransform(float potential_reward_scale, float ema_decay = 0.999f);

        void on_add(const torch::Tensor &single_step_reward) override;

        torch::Tensor transform(const torch::Tensor &batch_step_reward) override;

    private:
        float potential_reward_ema_decay_;
        float potential_reward_ema_mean_;
        float potential_reward_ema_var_;
        bool ema_initialized_;

        float potential_reward_scale;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_EMA_REPLAY_BUFFER_H
