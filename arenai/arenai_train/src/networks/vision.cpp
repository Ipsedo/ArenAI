//
// Created by samuel on 03/10/2025.
//

#include "./vision.h"

#include <arenai_core/constants.h>

ConvolutionNetwork::ConvolutionNetwork(
    const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums)
    : cnn(register_module("cnn", torch::nn::Sequential())) {

    int w = ENEMY_VISION_WIDTH, h = ENEMY_VISION_HEIGHT;

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

    output_height = h;
    output_width = w;
}

torch::Tensor ConvolutionNetwork::forward(const torch::Tensor &input) {
    if (input.dtype() != torch::kUInt8) throw std::runtime_error("Input must be UInt8");

    return cnn->forward(input.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f));
}

int ConvolutionNetwork::get_output_size() const { return output_size; }

int ConvolutionNetwork::get_output_width() const { return output_width; }

int ConvolutionNetwork::get_output_height() const { return output_height; }

/*
 * Conv Transpose
 */
TransposedConvolutionNetwork::TransposedConvolutionNetwork(
    const std::vector<std::tuple<int, int>> &encode_channels,
    const std::vector<int> &encode_group_norm_nums, const int encoded_image_height,
    const int encoded_image_width)
    : cnn(register_module("cnn", torch::nn::Sequential())) {

    std::vector channels(encode_channels);
    std::ranges::reverse(channels);

    for (auto &[a, b]: channels) { std::swap(a, b); }

    std::vector group_norm_nums(encode_group_norm_nums);
    group_norm_nums.erase(group_norm_nums.end());
    std::ranges::reverse(group_norm_nums);

    int w = encoded_image_width, h = encoded_image_height;

    for (int i = 0; i < channels.size(); i++) {
        const auto &[c_i, c_o] = channels[i];

        constexpr int padding = 1, stride = 2, kernel = 3, output_padding = 1;

        w = (w - 1) * stride - 2 * padding + kernel + output_padding;
        h = (h - 1) * stride - 2 * padding + kernel + output_padding;

        cnn->push_back(
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(c_i, c_o, kernel)
                                           .stride(stride)
                                           .padding(padding)
                                           .output_padding(output_padding)));

        if (i < channels.size() - 1) {
            cnn->push_back(
                torch::nn::GroupNorm(torch::nn::GroupNormOptions(group_norm_nums[i], c_o)));
            cnn->push_back(torch::nn::SiLU());
        } else cnn->push_back(torch::nn::Tanh());
    }
}

torch::Tensor TransposedConvolutionNetwork::forward_to_loss(
    const torch::Tensor &encoded_image, const torch::Tensor &vision_uint8) {
    const auto decoded_image = cnn->forward(encoded_image);
    const auto vision_float = vision_uint8.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f);

    return torch::mse_loss(decoded_image, vision_float);
}
