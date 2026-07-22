//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_AGENT_HOST_TRUNCATED_NORMAL_H
#define ARENAI_AGENT_HOST_TRUNCATED_NORMAL_H

#include <torch/torch.h>

namespace arenai::agent {

    constexpr float MIN_VALUE = -1.f;
    constexpr float MAX_VALUE = 1.f;

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

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_TRUNCATED_NORMAL_H
