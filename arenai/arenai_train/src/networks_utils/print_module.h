//
// Created by samuel on 14/02/2026.
//

#ifndef ARENAI_TRAIN_HOST_PRINT_MODULE_H
#define ARENAI_TRAIN_HOST_PRINT_MODULE_H

#include <torch/torch.h>

void dump_module_tree(
    const std::shared_ptr<torch::nn::Module> &m, std::ostream &out, int indent,
    const std::string &name);

#endif//ARENAI_TRAIN_HOST_PRINT_MODULE_H
