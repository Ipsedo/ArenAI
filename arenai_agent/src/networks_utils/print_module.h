//
// Created by samuel on 14/02/2026.
//

#ifndef ARENAI_AGENT_HOST_PRINT_MODULE_H
#define ARENAI_AGENT_HOST_PRINT_MODULE_H

#include <torch/torch.h>

namespace arenai::agent {

    void dump_module_tree(
        const std::shared_ptr<torch::nn::Module> &m, std::ostream &out, int indent,
        const std::string &name);

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_PRINT_MODULE_H
