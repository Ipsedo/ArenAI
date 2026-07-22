//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_AGENT_HOST_TARGET_UPDATE_H
#define ARENAI_AGENT_HOST_TARGET_UPDATE_H

#include <memory>

#include <torch/torch.h>

namespace arenai::agent {

    void hard_update(
        const std::shared_ptr<torch::nn::Module> &to,
        const std::shared_ptr<torch::nn::Module> &from);
    void soft_update(
        const std::shared_ptr<torch::nn::Module> &to,
        const std::shared_ptr<torch::nn::Module> &from, float tau);

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_TARGET_UPDATE_H
