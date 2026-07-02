//
// Created by samuel on 10/02/2026.
//

#ifndef ARENAI_TRAIN_HOST_GAUSSIAN_TANH_H
#define ARENAI_TRAIN_HOST_GAUSSIAN_TANH_H

#include <torch/torch.h>

namespace arenai::train {

    std::pair<torch::Tensor, torch::Tensor>
    gaussian_tanh_sample(const torch::Tensor &mu, const torch::Tensor &sigma);

    torch::Tensor gaussian_tanh_log_pdf(
        const torch::Tensor &u, const torch::Tensor &mu, const torch::Tensor &sigma);

    float gaussian_tanh_target_entropy(int nb_actions, float target_sigma, int nb_samples = 100000);

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_GAUSSIAN_TANH_H
