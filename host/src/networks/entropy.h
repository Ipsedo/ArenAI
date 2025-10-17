//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_ENTROPY_H
#define ARENAI_TRAIN_HOST_ENTROPY_H

#include <torch/torch.h>

class AlphaParameter final : public torch::nn::Module {
public:
    explicit AlphaParameter(float initial_alpha);

    torch::Tensor log_alpha();
    torch::Tensor alpha();

private:
    torch::Tensor log_alpha_tensor;
};

#endif//ARENAI_TRAIN_HOST_ENTROPY_H
