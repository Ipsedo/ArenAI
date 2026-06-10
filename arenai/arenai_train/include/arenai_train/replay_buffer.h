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

struct TorchInputStep {
    TorchState state;
    TorchAction action;
    torch::Tensor main_reward;
    torch::Tensor potential_reward;
    torch::Tensor done;
    TorchState next_state;
};

struct TorchOutputStep {
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

    TorchOutputStep sample(int batch_size, torch::Device device);

    void add(const TorchInputStep &step);

    int size() const;

protected:
    virtual void on_add_step(int write_idx, const TorchInputStep &step) = 0;
    virtual TorchOutputStep to_output_step(const TorchInputStep &batch_steps) = 0;

    bool is_full() const;

private:
    bool initialized_;

    size_t memory_size_;
    size_t write_idx_;
    size_t size_;

    torch::Tensor store_state_vision_;
    torch::Tensor store_state_proprioception_;
    torch::Tensor store_cont_action_;
    torch::Tensor store_disc_action_;
    torch::Tensor store_main_reward_;
    torch::Tensor store_potential_reward_;
    torch::Tensor store_done_;
    torch::Tensor store_next_vision_;
    torch::Tensor store_next_proprioception_;

    void initialize(const TorchInputStep &first_step);
};

#endif// ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
