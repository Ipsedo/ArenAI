//
// Created by samuel on 03/10/2025.
//

#include "./vision.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    ConvolutionNetwork::ConvolutionNetwork(
        const int vision_height, const int vision_width,
        const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums)
        : cnn(register_module("cnn", torch::nn::Sequential())) {

        int w = vision_width, h = vision_height;

        for (int i = 0; i < channels.size(); i++) {
            const auto &[c_i, c_o] = channels[i];
            const auto groups = group_norm_nums[i];

            constexpr int padding = 1, stride = 2, kernel = 3;

            w = (w - kernel + 2 * padding) / stride + 1;
            h = (h - kernel + 2 * padding) / stride + 1;

            cnn->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c_i, c_o, kernel)
                                                 .stride(stride)
                                                 .padding(padding)
                                                 .bias(false)));
            cnn->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, c_o)));
            cnn->push_back(torch::nn::SiLU());
        }

        output_size = w * h * std::get<1>(channels.back());

        cnn->push_back(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1).end_dim(-1)));
    }

    torch::Tensor ConvolutionNetwork::forward(const torch::Tensor &input) {
        TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be UInt8");

        return cnn->forward(input.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f));
    }

    int ConvolutionNetwork::get_output_size() const { return output_size; }

    /*
     * Residual
     */

    ResidualBlock::ResidualBlock(const int channels)
        : conv_block(register_module(
            "conv_block",
            torch::nn::Sequential(
                torch::nn::SiLU(),
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(channels, channels, 3).stride(1).padding(1)),
                torch::nn::SiLU(),
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(channels, channels, 3).stride(1).padding(1))))) {}

    torch::Tensor ResidualBlock::forward(const torch::Tensor &input) {
        return conv_block->forward(input) + input;
    }

    /*
     * Impala CNN
     */

    ImpalaConvolutionNetwork::ImpalaConvolutionNetwork(
        const int vision_height, const int vision_width,
        const std::vector<std::tuple<int, int>> &channels)
        : cnn(register_module("cnn", torch::nn::Sequential())) {
        int w = vision_width, h = vision_height;

        for (auto [c_i, c_o]: channels) {
            constexpr int padding = 1, stride = 2, kernel = 3;

            w = (w - kernel + 2 * padding) / stride + 1;
            h = (h - kernel + 2 * padding) / stride + 1;

            cnn->push_back(torch::nn::Conv2d(
                torch::nn::Conv2dOptions(c_i, c_o, kernel).stride(1).padding(padding)));
            cnn->push_back(torch::nn::MaxPool2d(
                torch::nn::MaxPool2dOptions(kernel).stride(stride).padding(padding)));
            cnn->push_back(std::make_shared<ResidualBlock>(c_o));
        }

        cnn->push_back(torch::nn::SiLU());
        cnn->push_back(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1).end_dim(-1)));

        output_size = w * h * std::get<1>(channels.back());
    }

    torch::Tensor ImpalaConvolutionNetwork::forward(const torch::Tensor &input) {
        TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be UInt8");

        return cnn->forward(input.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f));
    }

    int ImpalaConvolutionNetwork::get_output_size() const { return output_size; }
}// namespace arenai::agent
