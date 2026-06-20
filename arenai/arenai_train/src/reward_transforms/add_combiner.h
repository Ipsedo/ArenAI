//
// Created by samuel on 20/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_ADD_COMBINER_H
#define ARENAI_TRAIN_HOST_ADD_COMBINER_H

#include <arenai_train/reward_transform.h>

class AddCombiner : public AbstractRewardsCombiner {
public:
    torch::Tensor to_reward(const InputRewards &batch_rewards) override;
};

#endif//ARENAI_TRAIN_HOST_ADD_COMBINER_H
