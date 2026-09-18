//
// Created by samuel on 18/09/2026.
//

#ifndef ARENAI_AGENT_HOST_BERNOULLI_H
#define ARENAI_AGENT_HOST_BERNOULLI_H

#include <torch/torch.h>

namespace arenai::agent {

    torch::Tensor bernoulli_sample(const torch::Tensor &probabilities);
    torch::Tensor bernoulli_max_action(const torch::Tensor &probabilities);

    torch::Tensor
    bernoulli_log_proba(const torch::Tensor &actions, const torch::Tensor &probabilities);

    torch::Tensor bernoulli_entropy(const torch::Tensor &probabilities);

    float bernoulli_maximum_entropy();

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_BERNOULLI_H
