//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_REPLAY_BUFFER_H

#include <random>
#include <vector>

#include <torch/torch.h>

struct TorchState {
    torch::Tensor vision;
    torch::Tensor proprioception;
};

struct TorchStep {
    TorchState state;
    torch::Tensor action;
    torch::Tensor reward;
    torch::Tensor done;
    TorchState next_state;
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(int memory_size, int seed);

    TorchStep sample(int batch_size, torch::Device device);

    void add(TorchStep step);

private:
    std::mt19937 rng;
    int memory_size;
    std::vector<TorchStep> memory;
};

#endif// ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
