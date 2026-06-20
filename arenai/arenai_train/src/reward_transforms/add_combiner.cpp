//
// Created by samuel on 20/06/2026.
//

#include "./add_combiner.h"

torch::Tensor AddCombiner::to_reward(const InputRewards &batch_rewards) {
    return batch_rewards.main_reward + batch_rewards.potential_reward;
}
