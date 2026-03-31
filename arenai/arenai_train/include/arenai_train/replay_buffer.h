//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_REPLAY_BUFFER_H

#include <vector>

#include <torch/torch.h>

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
    explicit ReplayBuffer(int memory_size);

    TorchStep sample(int batch_size, torch::Device device) const;

    void add(const TorchStep &step);

    int size() const;

private:
    size_t memory_size_;
    size_t write_idx_;
    size_t size_;

    bool initialized_;

    torch::Tensor store_state_vision_;
    torch::Tensor store_state_proprioception_;
    torch::Tensor store_cont_action_;
    torch::Tensor store_disc_action_;
    torch::Tensor store_reward_;
    torch::Tensor store_done_;
    torch::Tensor store_next_vision_;
    torch::Tensor store_next_proprioception_;

    void initialize(const TorchStep &first_step);
};

#endif// ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
