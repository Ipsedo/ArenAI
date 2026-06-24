//
// Created by samuel on 20/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_REWARD_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_REWARD_REPLAY_BUFFER_H

#include <arenai_train/replay_buffer.h>

#include "../reward_transforms/reward_transform.h"

class RewardTransformReplayBuffer : public ReplayBuffer {
public:
    RewardTransformReplayBuffer(
        int memory_size, const std::shared_ptr<AbstractRewardTransform> &main_reward_transform,
        const std::shared_ptr<AbstractRewardTransform> &potential_reward_transform,
        const std::shared_ptr<AbstractRewardsCombiner> &combiner);

protected:
    void on_add_step(const TorchInputStep &single_step) const override;

    TorchOutputStep to_output(const TorchInputStep &batch_steps) const override;

private:
    std::shared_ptr<AbstractRewardTransform> main_reward_transform_;
    std::shared_ptr<AbstractRewardTransform> potential_reward_transform_;

    std::shared_ptr<AbstractRewardsCombiner> combiner_;
};

#endif//ARENAI_TRAIN_HOST_REWARD_REPLAY_BUFFER_H
