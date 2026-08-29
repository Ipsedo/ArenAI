//
// Created by samuel on 08/02/2026.
//

#ifndef ARENAI_AGENT_HOST_BETA_LAW_H
#define ARENAI_AGENT_HOST_BETA_LAW_H

#include <torch/torch.h>

namespace arenai::agent {

    // Beta distribution rescaled to the [-1, 1] action support

    torch::Tensor beta_law_sample(const torch::Tensor &alpha, const torch::Tensor &beta);
    torch::Tensor beta_law_log_proba(
        const torch::Tensor &x, const torch::Tensor &alpha, const torch::Tensor &beta);
    torch::Tensor beta_law_entropy(const torch::Tensor &alpha, const torch::Tensor &beta);

    torch::Tensor beta_law_mean_action(const torch::Tensor &alpha, const torch::Tensor &beta);

    float beta_law_target_entropy(const int &nb_actions);

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_BETA_LAW_H
