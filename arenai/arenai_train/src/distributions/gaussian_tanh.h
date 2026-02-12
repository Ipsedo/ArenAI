//
// Created by samuel on 10/02/2026.
//

#ifndef ARENAI_TRAIN_HOST_GAUSSIAN_TANH_H
#define ARENAI_TRAIN_HOST_GAUSSIAN_TANH_H

#include <torch/torch.h>

torch::Tensor gaussian_tanh_sample(const torch::Tensor &mu, const torch::Tensor &sigma);

torch::Tensor
gaussian_tanh_log_pdf(const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma);

#endif//ARENAI_TRAIN_HOST_GAUSSIAN_TANH_H
