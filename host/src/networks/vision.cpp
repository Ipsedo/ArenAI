//
// Created by samuel on 03/10/2025.
//

#include "./vision.h"

ConvolutionNetwork::ConvolutionNetwork()
    : cnn(register_module(
        "cnn", torch::nn::Sequential(
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 8, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 48, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(48, 64, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 96, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 128, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   // 1 * 1 * 128 -> 128
                   torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1).end_dim(-1))))) {}

torch::Tensor ConvolutionNetwork::forward(const torch::Tensor &input) {
    return cnn->forward(input);
}
