//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_TRUNCATED_NORMAL_H
#define ARENAI_TRAIN_HOST_TRUNCATED_NORMAL_H

#include <torch/torch.h>

torch::Tensor truncated_normal_sample(
    const torch::Tensor &mu, const torch::Tensor &sigma, float min_value, float max_value);

torch::Tensor truncated_normal_log_pdf(
    const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma, float min_value,
    float max_value);

torch::Tensor truncated_normal_entropy(
    const torch::Tensor &mu, const torch::Tensor &sigma, float min_value, float max_value);

float get_truncated_normal_target_entropy(int nb_actions, float min_value, float max_value);

torch::Tensor gaussian_tanh_sample(const torch::Tensor &mu, const torch::Tensor &sigma);
torch::Tensor
gaussian_tanh_log_pdf(const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma);

#endif//ARENAI_TRAIN_HOST_TRUNCATED_NORMAL_H
