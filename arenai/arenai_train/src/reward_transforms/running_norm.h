//
// Created by samuel on 10/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_POTENTIAL_REWARD_EMA_REPLAY_BUFFER_H
#define ARENAI_TRAIN_HOST_POTENTIAL_REWARD_EMA_REPLAY_BUFFER_H

#include "./reward_transform.h"

class NormalizedPotentialTransform : public AbstractRewardTransform {
public:
    NormalizedPotentialTransform(int memory_size, float potential_reward_scale);

    torch::Tensor transform(const torch::Tensor &single_step_rewards) override;

private:
    size_t memory_size_;
    size_t write_idx_;
    size_t size_;

    float potential_reward_scale_;

    double potential_running_sum_;
    double potential_running_sum_sq_;

    std::vector<double> potential_reward_history_;

    bool is_full() const;
};

class NormalizedNonZeroTransform : public AbstractRewardTransform {
public:
    explicit NormalizedNonZeroTransform(int memory_size);

    torch::Tensor transform(const torch::Tensor &single_step_reward) override;

private:
    size_t memory_size_;
    size_t write_idx_;
    size_t size_;
    size_t non_zero_nb_;

    double non_zero_running_sum_sq_;

    std::vector<double> reward_history_;

    bool is_full() const;
};

#endif//ARENAI_TRAIN_HOST_POTENTIAL_REWARD_EMA_REPLAY_BUFFER_H
