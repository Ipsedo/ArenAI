//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_VISION_H
#define ARENAI_TRAIN_HOST_VISION_H

#include <torch/torch.h>

class ConvolutionNetwork final : public torch::nn::Module {
public:
    ConvolutionNetwork();

    torch::Tensor forward(const torch::Tensor &input);

private:
    torch::nn::Sequential cnn{nullptr};
};

#endif// ARENAI_TRAIN_HOST_VISION_H
