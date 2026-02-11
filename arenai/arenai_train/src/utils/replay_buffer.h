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

struct TorchStep {
    TorchState state;
    torch::Tensor action;
    torch::Tensor log_proba;
    torch::Tensor reward;
    torch::Tensor done;
    TorchState next_state;
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(int memory_size);

    TorchStep sample(int batch_size, torch::Device device);

    void add(const TorchStep &step);

    int size() const;

private:
    size_t memory_size_;
    size_t write_idx_;
    size_t size_;
    std::vector<TorchStep> memory;

    static TorchStep clone_step(const TorchStep &to_clone);
};

#endif// ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
