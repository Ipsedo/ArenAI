//
// Created by samuel on 10/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_POTENTIAL_REWARD_EMA_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_POTENTIAL_REWARD_EMA_REPLAY_BUFFER_H

#include <arenai_train/replay_buffer.h>

class NormalizedPotentialRewardReplayBuffer : public ReplayBuffer {
public:
    NormalizedPotentialRewardReplayBuffer(int memory_size, float potential_reward_scale);

protected:
    TorchInputStep on_add_step(int write_idx, const TorchInputStep &step) override;
    TorchOutputStep to_output_step(const TorchInputStep &batch_steps) override;

private:
    float potential_reward_scale_;

    double running_sum_;
    double running_sum_sq_;

    std::vector<double> reward_history_;
};

#endif//ARENAI_TRAIN_HOST_POTENTIAL_REWARD_EMA_REPLAY_BUFFER_H
