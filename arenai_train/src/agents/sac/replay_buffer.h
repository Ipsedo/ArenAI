//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_REPLAY_BUFFER_H

#include <vector>

#include <torch/torch.h>

#include "../torch_types.h"

namespace arenai::train {

    struct SacInputStep {
        TorchState state;
        TorchAction action;
        torch::Tensor reward;
        torch::Tensor done;
        torch::Tensor truncated;
    };

    struct SacTrainStep {
        TorchState state;
        TorchAction action;
        torch::Tensor reward;
        torch::Tensor done;
        TorchState next_state;
    };

    class SacReplayBuffer {
    public:
        virtual ~SacReplayBuffer() = default;

        explicit SacReplayBuffer(int memory_size);

        SacTrainStep sample(int batch_size, torch::Device device) const;

        void add(const SacInputStep &step);
        void finish_episode(const TorchState &final_step);

        size_t size() const;

    private:
        void initialize(const SacInputStep &first_step);
        void advance_write_idx();

        bool initialized_;

        // total transition budget: the ring holds memory_size_ / nb_tanks_ steps
        size_t memory_size_;
        size_t nb_steps_;
        size_t write_idx_;
        size_t size_;

        int64_t nb_tanks_;

        torch::Tensor store_vision_;
        torch::Tensor store_proprioception_;
        torch::Tensor store_cont_action_;
        torch::Tensor store_disc_action_;
        torch::Tensor store_reward_;
        torch::Tensor store_done_;

        // [mem, nb_tanks] whether the (step, tank) pair can start a sampled transition
        torch::Tensor store_sampleable_;
        // [nb_tanks] tanks already done/truncated in the current episode
        torch::Tensor already_terminated_;
    };

}// namespace arenai::train

#endif// ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
