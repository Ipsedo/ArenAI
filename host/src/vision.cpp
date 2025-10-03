//
// Created by samuel on 03/10/2025.
//

#include "./vision.h"

ConvolutionNetwork::ConvolutionNetwork()
    : cnn(register_module(
          "cnn",
          torch::nn::Sequential(
              // 1 : 128 -> 64
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(3, 8, 3).padding(1).stride(2)),
              torch::nn::Mish(), torch::nn::InstanceNorm1d(8),
              // 2 : 64 -> 32
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(8, 16, 3).padding(1).stride(2)),
              torch::nn::Mish(), torch::nn::InstanceNorm1d(16),
              // 3 : 32 -> 16
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(16, 32, 3).padding(1).stride(2)),
              torch::nn::Mish(), torch::nn::InstanceNorm1d(32),
              // 4 : 16 -> 8
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(32, 64, 3).padding(1).stride(2)),
              torch::nn::Mish(), torch::nn::InstanceNorm1d(64),
              // 4 : 8 -> 4
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(64, 128, 3).padding(1).stride(2)),
              torch::nn::Mish(), torch::nn::InstanceNorm1d(128),
              // 5 : 4 -> 2
              torch::nn::Conv2d(
                  torch::nn::Conv2dOptions(128, 256, 3).padding(1).stride(2)),
              torch::nn::Mish(), torch::nn::InstanceNorm1d(256),
              // 2 * 2 * 256 -> 1024
              torch::nn::Flatten(
                  torch::nn::FlattenOptions().start_dim(1).end_dim(-1))))) {}

torch::Tensor ConvolutionNetwork::forward(torch::Tensor input) {
  return cnn->forward(input);
}
