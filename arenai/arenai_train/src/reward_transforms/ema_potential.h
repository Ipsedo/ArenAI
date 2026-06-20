//
// Created by samuel on 10/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_EMA_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_EMA_REPLAY_BUFFER_H

#include <arenai_train/reward_transform.h>

class EmaPotentialTransform : public AbstractRewardsTransform {
public:
    EmaPotentialTransform(float potential_reward_scale, float ema_decay = 0.999f);

    InputRewards transform(const InputRewards &single_step_rewards) override;

private:
    float potential_reward_ema_decay_;
    float potential_reward_ema_mean_;
    float potential_reward_ema_var_;
    bool ema_initialized_;

    float potential_reward_scale;
};

#endif//ARENAI_TRAIN_HOST_EMA_REPLAY_BUFFER_H
