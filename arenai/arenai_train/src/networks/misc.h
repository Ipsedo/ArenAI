//
// Created by samuel on 19/05/2026.
//

#ifndef ARENAI_TRAIN_HOST_MISC_H
#define ARENAI_TRAIN_HOST_MISC_H

#include <torch/torch.h>

class Clamp : public torch::nn::Module {
public:
    Clamp(float lower_bound, float upper_bound);

    torch::Tensor forward(const torch::Tensor &x);

private:
    float lower_bound;
    float upper_bound;
};

class Exp : public torch::nn::Module {
public:
    torch::Tensor forward(const torch::Tensor &x);
};

#endif//ARENAI_TRAIN_HOST_MISC_H
