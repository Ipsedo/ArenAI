//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_REPLAY_BUFFER_H

#include <vector>

#include <torch/torch.h>

#include "reward_transform.h"

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

    explicit ReplayBuffer(
        int memory_size,
        const std::vector<std::shared_ptr<AbstractRewardsTransform>> &rewards_transforms,
        const std::shared_ptr<AbstractRewardsCombiner> &rewards_combiner);

    TorchOutputStep sample(int batch_size, torch::Device device) const;

    void add(const TorchInputStep &step);

    int size() const;

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

    std::vector<std::shared_ptr<AbstractRewardsTransform>> rewards_transforms_;
    std::shared_ptr<AbstractRewardsCombiner> rewards_combiner_;

    void initialize(const TorchInputStep &first_step);

    bool is_full() const;
};

#endif// ARENAI_TRAIN_HOST_REPLAY_BUFFER_H
