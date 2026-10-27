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

    // one optimizer step on the gradients accumulated by the minibatches' backward passes
    void clipped_step(
        torch::optim::Optimizer &optim, const std::vector<torch::Tensor> &parameters,
        float grad_norm_max);

}// namespace arenai::agent

#endif//ARENAI_PPO_UPDATE_H
