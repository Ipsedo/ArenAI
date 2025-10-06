//
// Created by samuel on 03/10/2025.
//

#include "./vision.h"

ConvolutionNetwork::ConvolutionNetwork()
    : cnn(register_module(
        "cnn", torch::nn::Sequential(
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1).stride(2)),
                   torch::nn::Mish(), torch::nn::InstanceNorm1d(16),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 24, 3).padding(1).stride(2)),
                   torch::nn::Mish(), torch::nn::InstanceNorm1d(24),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(24, 32, 3).padding(1).stride(2)),
                   torch::nn::Mish(), torch::nn::InstanceNorm1d(32),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 48, 3).padding(1).stride(2)),
                   torch::nn::Mish(), torch::nn::InstanceNorm1d(48),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(48, 64, 3).padding(1).stride(2)),
                   torch::nn::Mish(), torch::nn::InstanceNorm1d(64),
                   torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 96, 3).padding(1).stride(2)),
                   torch::nn::Mish(), torch::nn::InstanceNorm1d(96),
                   // 2 * 2 * 96 -> 384
                   torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1).end_dim(-1))))) {}

torch::Tensor ConvolutionNetwork::forward(torch::Tensor input) { return cnn->forward(input); }
