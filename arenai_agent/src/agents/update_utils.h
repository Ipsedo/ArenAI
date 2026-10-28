//
// Created by claude on 27/10/2026.
//

#ifndef ARENAI_PPO_UPDATE_H
#define ARENAI_PPO_UPDATE_H

#include <vector>

#include <torch/torch.h>

namespace arenai::agent {

    // KL estimate of a whole update from its per-row values, the 1 % largest trimmed
    float trimmed_kl(const std::vector<torch::Tensor> &kl_per_rows);

    // a shuffled epoch over units: one optimizer batch per batch_size units, each cut into
    // device blocks of at most gpu_batch_size units (the forward / backward memory bound)
    std::vector<std::vector<torch::Tensor>>
    shuffled_batches(const torch::Tensor &units, int64_t batch_size, int64_t gpu_batch_size);

}// namespace arenai::agent

#endif//ARENAI_PPO_UPDATE_H
