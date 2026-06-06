//
// Created by samuel on 03/10/2025.
//

#include "./vision.h"

#include "arenai_core/constants.h"

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

        cnn->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c_i, c_o, kernel)
                                             .stride(stride)
                                             .padding(padding)
                                             .bias(false)));
        cnn->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, c_o)));
        cnn->push_back(torch::nn::GELU());
    }

    output_size = w * h * std::get<1>(channels.back());
}

torch::Tensor ConvolutionNetwork::forward(const torch::Tensor &input) {
    return cnn->forward(input);
}

int ConvolutionNetwork::get_output_size() const { return output_size; }

// Decoder
TransposedConvolutionNetwork::TransposedConvolutionNetwork(
    const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums)
    : tr_cnn(register_module("tr_cnn", torch::nn::Sequential())) {

    for (int i = channels.size() - 1; i >= 0; i--) {
        const auto &[c_o, c_i] = channels[i];

        tr_cnn->push_back(torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(c_i, c_o, 3).stride(2).padding(1).output_padding(1)));

        if (i != 0) {
            const auto groups = group_norm_nums[i - 1];

            tr_cnn->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, c_o)));
            tr_cnn->push_back(torch::nn::GELU());
        } else {
            tr_cnn->push_back(torch::nn::Tanh());
        }
    }
}

torch::Tensor TransposedConvolutionNetwork::forward(const torch::Tensor &input) {
    return tr_cnn->forward(input);
}

// Auto encoder

VisionAutoEncoder::VisionAutoEncoder(
    const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums)
    : encoder(register_module(
        "encoder", std::make_shared<ConvolutionNetwork>(channels, group_norm_nums))),
      decoder(register_module(
          "decoder", std::make_shared<TransposedConvolutionNetwork>(channels, group_norm_nums))) {}

vision_output VisionAutoEncoder::forward(const torch::Tensor &input) const {
    if (input.dtype() != torch::kUInt8) throw std::runtime_error("Input must be UInt8");
    const auto input_float = input.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f);

    const auto encoded_image = encoder->forward(input_float);
    const auto decoded_image = decoder->forward(encoded_image);

    return {
        encoded_image.flatten(1, -1),
        torch::mse_loss(input_float, decoded_image, at::Reduction::None)};
}

int VisionAutoEncoder::get_output_size() const { return encoder->get_output_size(); }
