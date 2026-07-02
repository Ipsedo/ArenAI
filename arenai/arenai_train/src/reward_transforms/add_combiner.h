//
// Created by samuel on 20/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_ADD_COMBINER_H
#define ARENAI_TRAIN_HOST_ADD_COMBINER_H

#include "./reward_transform.h"

namespace arenai::train {

    class AddCombiner : public AbstractRewardsCombiner {
    public:
        torch::Tensor to_reward(const InputRewards &batch_rewards) override;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_ADD_COMBINER_H
