//
// Created by samuel on 03/10/2025.
//

#ifndef PHYVR_TRAIN_HOST_VISION_H
#define PHYVR_TRAIN_HOST_VISION_H

#include <torch/torch.h>

class ConvolutionNetwork final : public torch::nn::Module {
public:
    ConvolutionNetwork();

    torch::Tensor forward(torch::Tensor input);

private:
    torch::nn::Sequential cnn{nullptr};
};

#endif// PHYVR_TRAIN_HOST_VISION_H
