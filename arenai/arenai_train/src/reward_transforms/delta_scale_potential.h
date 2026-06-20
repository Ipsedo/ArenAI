//
// Created by samuel on 12/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_DELTA_SCALE_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_DELTA_SCALE_REPLAY_BUFFER_H

#include <arenai_train/reward_transform.h>

class DeltaScalePotentialRewardTransform : public AbstractRewardsTransform {
public:
    DeltaScalePotentialRewardTransform(float wanted_frequency, float target_potential_reward);

    InputRewards transform(const InputRewards &single_step_rewards) override;

private:
    float potential_reward_scale_;

    static float compute_potential_reward_scale(
        float wanted_frequency, float distance_scale, float target_reward, float typical_fraction);
};

#endif//ARENAI_TRAIN_HOST_DELTA_SCALE_REPLAY_BUFFER_H
