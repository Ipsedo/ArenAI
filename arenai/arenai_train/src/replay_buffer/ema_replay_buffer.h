//
// Created by samuel on 10/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_EMA_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_EMA_REPLAY_BUFFER_H

#include <arenai_train/replay_buffer.h>

class PotentialRewardEmaReplayBuffer : public ReplayBuffer {
public:
    PotentialRewardEmaReplayBuffer(
        int memory_size, float potential_reward_scale, float ema_decay = 0.999);

protected:
    TorchInputStep on_add_step(int write_idx, const TorchInputStep &step) override;

    TorchOutputStep to_output_step(const TorchInputStep &batch_steps) override;

private:
    float potential_reward_ema_decay_;
    float potential_reward_ema_mean_;
    float potential_reward_ema_var_;
    bool ema_initialized_;

    float potential_reward_scale;
};

#endif//ARENAI_TRAIN_HOST_EMA_REPLAY_BUFFER_H
