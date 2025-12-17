//
// Created by samuel on 02/12/2025.
//

#ifndef ARENAI_TRAIN_HOST_GUMBEL_SOFTMAX_H
#define ARENAI_TRAIN_HOST_GUMBEL_SOFTMAX_H

#include <torch/torch.h>

class GumbelSoftmax : public torch::nn::Module {
public:
    explicit GumbelSoftmax(const int &dim, const float &tau = 1.f, const float &epsilon = 1e-20f);

    torch::Tensor forward(const torch::Tensor &x);

private:
    int dim;
    float tau;
    float epsilon;
};

#endif//ARENAI_TRAIN_HOST_GUMBEL_SOFTMAX_H
