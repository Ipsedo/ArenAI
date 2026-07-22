//
// Created by samuel on 24/10/2025.
//

#ifndef ARENAI_AGENT_HOST_INIT_H
#define ARENAI_AGENT_HOST_INIT_H

#include <torch/torch.h>

namespace arenai::agent {

    void init_hidden_weights(torch::nn::Module &module);

    void init_mu_output_weights(torch::nn::Module &module);
    void init_sigma_output_weights(torch::nn::Module &module);
    void init_discrete_output_weights(torch::nn::Module &module);

    void init_value_output_weights(torch::nn::Module &module);

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_INIT_H
