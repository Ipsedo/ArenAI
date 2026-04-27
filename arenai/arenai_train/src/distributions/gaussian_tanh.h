//
// Created by samuel on 10/02/2026.
//

#ifndef ARENAI_TRAIN_HOST_GAUSSIAN_TANH_H
#define ARENAI_TRAIN_HOST_GAUSSIAN_TANH_H

#include <torch/torch.h>

std::pair<torch::Tensor, torch::Tensor>
gaussian_tanh_sample(const torch::Tensor &mu, const torch::Tensor &sigma);

torch::Tensor
gaussian_tanh_log_pdf(const torch::Tensor &u, const torch::Tensor &mu, const torch::Tensor &sigma);

float gaussian_tanh_target_entropy(int nb_actions, float target_sigma);

#endif//ARENAI_TRAIN_HOST_GAUSSIAN_TANH_H
