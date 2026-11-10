#ifndef ARENAI_SAC_REPLAY_BUFFER_H
#define ARENAI_SAC_REPLAY_BUFFER_H

#include <cstdint>

#include <torch/torch.h>

#include "../torch_types.h"

namespace arenai::agent {

    struct SacInputStep {
        TorchState state;
        TorchAction action;
        torch::Tensor reward;
        torch::Tensor done;
        torch::Tensor truncated;
    };

    // [B, ...]
    struct SacBatch {
        TorchState states;
        TorchAction actions;
        torch::Tensor rewards;
        // [B, 1]
        torch::Tensor cuts;
        TorchState next_states;
    };

    class SacReplayBuffer {
    public:
        SacReplayBuffer(int64_t capacity, float armed_fraction);

        void add(const SacInputStep &step);
        void finish_episode(const TorchState &final_state);

        int64_t nb_transitions() const;

        SacBatch sample(int64_t batch_size, torch::Device device) const;

    private:
        int64_t capacity;
        float armed_fraction;

        int64_t nb_tanks = 0;
        int64_t slots_per_tank = 0;
        int64_t write_index = 0;
        int64_t nb_written = 0;

        // [nb_tanks, slots_per_tank, ...]
        torch::Tensor visions;
        torch::Tensor states;
        torch::Tensor fire_masks;
        torch::Tensor continuous_actions;
        torch::Tensor discrete_actions;
        torch::Tensor rewards;
        torch::Tensor cuts;
        // [nb_tanks, slots_per_tank]
        torch::Tensor valids;
        // [nb_tanks, slots_per_tank]
        torch::Tensor armed;

        // [nb_tanks]
        torch::Tensor already_terminated;

        void allocate(const SacInputStep &step);

        void write_observation(const TorchState &state, const torch::Tensor &valid);

        // [nb_tanks, slots_per_tank]
        torch::Tensor sampleable() const;

        static torch::Tensor draw(const torch::Tensor &candidates, int64_t nb);
    };

}// namespace arenai::agent

#endif//ARENAI_SAC_REPLAY_BUFFER_H
