//
// Created by claude on 08/11/2026.
//

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
        // segment end that is not a death: cuts the value target like a done
        torch::Tensor truncated;
    };

    // a random batch, every tensor [B, ...] on the requested device
    struct SacBatch {
        TorchState states;
        TorchAction actions;
        torch::Tensor rewards;
        // [B, 1] 1 when the transition bootstraps on nothing (death or truncation)
        torch::Tensor cuts;
        TorchState next_states;
    };

    // Off-policy replay stored on the CPU as one ring per tank, the observations in uint8
    // and written once: the next state of slot t is slot t + 1 of the same tank. A slot is
    // sampleable when it holds a live transition and its successor has been written, so
    // the newest slot never is. finish_episode() writes the closing observation in a slot
    // of its own (no transition) so the last step of an episode gets its next state.
    class SacReplayBuffer {
    public:
        // capacity: transitions over all the tanks, split evenly between them
        explicit SacReplayBuffer(int64_t capacity);

        void add(const SacInputStep &step);
        void finish_episode(const TorchState &final_state);

        // number of sampleable transitions
        int64_t nb_transitions() const;

        SacBatch sample(int64_t batch_size, torch::Device device) const;

    private:
        int64_t capacity;

        int64_t nb_tanks = 0;
        int64_t slots_per_tank = 0;
        // next slot to write, shared by every tank (they step together)
        int64_t write_index = 0;
        int64_t nb_written = 0;

        // [nb_tanks, slots_per_tank, ...]
        torch::Tensor visions;
        torch::Tensor proprioceptions;
        torch::Tensor fire_masks;
        torch::Tensor continuous_actions;
        torch::Tensor discrete_actions;
        torch::Tensor rewards;
        torch::Tensor cuts;
        // [nb_tanks, slots_per_tank] whether the slot holds a live transition
        torch::Tensor valids;

        // [nb_tanks] tanks already dead in the current episode
        torch::Tensor already_terminated;

        void allocate(const SacInputStep &step);

        // writes the observation at write_index and advances it; a slot written with
        // valid = false is an observation only (its action / reward are ignored)
        void write_observation(const TorchState &state, const torch::Tensor &valid);

        // [nb_tanks, slots_per_tank] slots that can be sampled
        torch::Tensor sampleable() const;
    };

}// namespace arenai::agent

#endif//ARENAI_SAC_REPLAY_BUFFER_H
