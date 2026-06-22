//
// Created by samuel on 12/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_DELTA_SCALE_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_DELTA_SCALE_REPLAY_BUFFER_H

#include "./scale_potential.h"

class DeltaScalePotentialRewardTransform : public ScalePotentialTransform {
public:
    DeltaScalePotentialRewardTransform(float wanted_frequency, float target_potential_reward);

private:
    static float compute_potential_reward_scale(
        float wanted_frequency, float distance_scale, float target_reward, float typical_fraction);
};

#endif//ARENAI_TRAIN_HOST_DELTA_SCALE_REPLAY_BUFFER_H
