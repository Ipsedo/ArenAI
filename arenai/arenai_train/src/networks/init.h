//
// Created by samuel on 24/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_INIT_H
#define ARENAI_TRAIN_HOST_INIT_H

#include <torch/torch.h>

void init_weights(torch::nn::Module &module);

#endif //ARENAI_TRAIN_HOST_INIT_H