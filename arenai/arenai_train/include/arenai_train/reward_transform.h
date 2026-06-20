//
// Created by samuel on 20/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_REWARD_TRANSFORM_H
#define ARENAI_TRAIN_HOST_REWARD_TRANSFORM_H

#include <torch/torch.h>

struct InputRewards {
    torch::Tensor main_reward;
    torch::Tensor potential_reward;
};

class AbstractRewardsTransform {
public:
    virtual ~AbstractRewardsTransform() = default;

    virtual InputRewards transform(const InputRewards &single_step_rewards) = 0;
};

class AbstractRewardsCombiner {
public:
    virtual ~AbstractRewardsCombiner() = default;

    virtual torch::Tensor to_reward(const InputRewards &batch_rewards) = 0;
};

#endif//ARENAI_TRAIN_HOST_REWARD_TRANSFORM_H
