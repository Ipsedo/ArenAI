//
// Created by samuel on 03/10/2025.
//

#include "./vision.h"

#include "arenai_core/constants.h"

ConvolutionNetwork::ConvolutionNetwork(
    const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums)
    : cnn(register_module("cnn", torch::nn::Sequential())) {

    int w = ENEMY_VISION_SIZE, h = ENEMY_VISION_SIZE;

    for (int i = 0; i < channels.size(); i++) {
        const auto &[c_i, c_o] = channels[i];
        const auto groups = group_norm_nums[i];

        constexpr int padding = 1, stride = 2, kernel = 3;

        w = (w - kernel + 2 * padding) / stride + 1;
        h = (h - kernel + 2 * padding) / stride + 1;

        cnn->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(c_i, c_o, kernel).stride(stride).padding(padding)));
        cnn->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, c_o)));
        cnn->push_back(torch::nn::SiLU());
    }

    output_size = w * h * std::get<1>(channels.back());

    cnn->push_back(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1).end_dim(-1)));
}

torch::Tensor ConvolutionNetwork::forward(const torch::Tensor &input) {
    if (input.dtype() != torch::kUInt8) throw std::runtime_error("Input must be UInt8");

    return cnn->forward(input.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f));
}

int ConvolutionNetwork::get_output_size() const { return output_size; }
