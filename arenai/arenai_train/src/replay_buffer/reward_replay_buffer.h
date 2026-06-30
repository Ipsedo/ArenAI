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
        int memory_size, const std::shared_ptr<AbstractRewardTransform> &reward_transform,
        const std::shared_ptr<AbstractRewardTransform> &potential_transform);

protected:
    void on_add_step(const TorchInputStep &single_step) const override;

    TorchOutputStep transform_at_sample(const TorchInputStep &batch_steps) const override;

private:
    std::shared_ptr<AbstractRewardTransform> reward_transform_;
    std::shared_ptr<AbstractRewardTransform> potential_transform_;
};

#endif//ARENAI_TRAIN_HOST_REWARD_REPLAY_BUFFER_H
