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
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 96, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 128, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1).stride(2)),
                   torch::nn::SiLU(),
                   // 1 * 1 * 256
                   torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1).end_dim(-1)),
                   torch::nn::LayerNorm(torch::nn::LayerNormOptions({256}))))) {}

torch::Tensor ConvolutionNetwork::forward(const torch::Tensor &input) {
    if (input.dtype() != torch::kUInt8) throw std::runtime_error("Input must be UInt8");

    return cnn->forward(input.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f));
}
