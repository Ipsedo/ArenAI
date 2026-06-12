//
// Created by samuel on 12/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_DELTA_SCALE_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_DELTA_SCALE_REPLAY_BUFFER_H

#include <arenai_train/replay_buffer.h>

class DeltaScalePotentialRewardReplayBuffer : public ReplayBuffer {
public:
    DeltaScalePotentialRewardReplayBuffer(
        int memory_size, float wanted_frequency, float target_potential_reward);

protected:
    TorchInputStep on_add_step(int write_idx, const TorchInputStep &step) override;

    TorchOutputStep to_output_step(const TorchInputStep &batch_steps) override;

private:
    float potential_reward_scale_;

    static float compute_potential_reward_scale(
        float wanted_frequency, float distance_scale, float target_reward, float typical_fraction);
};

#endif//ARENAI_TRAIN_HOST_DELTA_SCALE_REPLAY_BUFFER_H
