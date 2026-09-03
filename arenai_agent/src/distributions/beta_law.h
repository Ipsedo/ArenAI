//
// Created by samuel on 08/02/2026.
//

#ifndef ARENAI_AGENT_HOST_BETA_LAW_H
#define ARENAI_AGENT_HOST_BETA_LAW_H

#include <torch/torch.h>

namespace arenai::agent {

    constexpr float KAPPA_SCALE = 300.f;
    // kappa floor = uniform law (alpha + beta = 2), asymptotically reachable
    constexpr float MIN_KAPPA_SIGMOID = 2.f / KAPPA_SCALE;

    // Beta distribution rescaled to the [-1, 1] action support

    torch::Tensor beta_law_sample(const torch::Tensor &mu, const torch::Tensor &kappa);
    torch::Tensor
    beta_law_log_proba(const torch::Tensor &x, const torch::Tensor &mu, const torch::Tensor &kappa);
    torch::Tensor beta_law_entropy(const torch::Tensor &mu, const torch::Tensor &kappa);

    torch::Tensor beta_law_mean_action(const torch::Tensor &mu, const torch::Tensor &kappa);

    float beta_law_target_entropy(const int &nb_actions, float target_concentration);

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_BETA_LAW_H
