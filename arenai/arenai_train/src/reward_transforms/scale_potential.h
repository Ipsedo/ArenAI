//
// Created by samuel on 11/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_SCALE_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_SCALE_REPLAY_BUFFER_H

#include <arenai_train/reward_transform.h>

class ScalePotentialTransform : public AbstractRewardsTransform {
public:
    explicit ScalePotentialTransform(float potential_reward_scale);

    InputRewards transform(const InputRewards &single_step_rewards) override;

private:
    float potential_reward_scale_;
};

#endif//ARENAI_TRAIN_HOST_SCALE_REPLAY_BUFFER_H
