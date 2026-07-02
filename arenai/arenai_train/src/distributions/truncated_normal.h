//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_TRUNCATED_NORMAL_H
#define ARENAI_TRAIN_HOST_TRUNCATED_NORMAL_H

#include <torch/torch.h>

#define MIN_VALUE -1.f
#define MAX_VALUE 1.f

namespace arenai::train {

    torch::Tensor truncated_normal_sample(
        const torch::Tensor &mu, const torch::Tensor &sigma, float min_value = MIN_VALUE,
        float max_value = MAX_VALUE);

    torch::Tensor truncated_normal_pdf(
        const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma,
        float min_value = MIN_VALUE, float max_value = MAX_VALUE);

    torch::Tensor truncated_normal_log_pdf(
        const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &sigma,
        float min_value = MIN_VALUE, float max_value = MAX_VALUE);

    torch::Tensor truncated_normal_entropy(
        const torch::Tensor &mu, const torch::Tensor &sigma, float min_value = MIN_VALUE,
        float max_value = MAX_VALUE);

    float truncated_normal_target_entropy(
        int nb_actions, float sigma, float min_value = MIN_VALUE, float max_value = MAX_VALUE);

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_TRUNCATED_NORMAL_H
