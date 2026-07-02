//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_REPLAY_BUFFER_H

#include <vector>

#include <torch/torch.h>

namespace arenai::train {

    struct TorchState {
        torch::Tensor vision;
        torch::Tensor proprioception;
    };

    struct TorchAction {
        torch::Tensor continuous_action;
        torch::Tensor discrete_action;
    };

    struct TorchStep {
        TorchState state;
        TorchAction action;
        torch::Tensor reward;
        torch::Tensor done;
        TorchState next_state;
    };

    class ReplayBuffer {
    public:
        virtual ~ReplayBuffer() = default;

        explicit ReplayBuffer(int memory_size);

        TorchStep sample(int batch_size, torch::Device device) const;

        void add(const TorchStep &step);

        size_t size() const;

    protected:
        virtual void on_add_step(const TorchStep &single_step) const;
        virtual TorchStep transform_at_sample(const TorchStep &batch_steps) const;

    private:
        bool initialized_;

        size_t memory_size_;
        size_t write_idx_;
        size_t size_;

        torch::Tensor store_state_vision_;
        torch::Tensor store_state_proprioception_;
        torch::Tensor store_cont_action_;
        torch::Tensor store_disc_action_;
        torch::Tensor store_reward_;
        torch::Tensor store_done_;
        torch::Tensor store_next_vision_;
        torch::Tensor store_next_proprioception_;

        void initialize(const TorchStep &first_step);

        bool is_full() const;
    };

}// namespace arenai::train

#endif// ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
