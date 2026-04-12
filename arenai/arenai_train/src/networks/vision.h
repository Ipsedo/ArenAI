//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_VISION_H
#define ARENAI_TRAIN_HOST_VISION_H

#include <torch/torch.h>

class ConvolutionNetwork final : public torch::nn::Module {
public:
    explicit ConvolutionNetwork(
        const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums);

    torch::Tensor forward(const torch::Tensor &input);

    int get_output_size() const;

    int get_output_width() const;
    int get_output_height() const;

private:
    torch::nn::Sequential cnn{nullptr};
    int output_size;

    int output_width, output_height;
};

class TransposedConvolutionNetwork final : public torch::nn::Module {
public:
    explicit TransposedConvolutionNetwork(
        const std::vector<std::tuple<int, int>> &encode_channels,
        const std::vector<int> &encode_group_norm_nums, int encoded_image_height,
        int encoded_image_width);

    torch::Tensor
    forward_to_loss(const torch::Tensor &encoded_image, const torch::Tensor &vision_uint8);

private:
    torch::nn::Sequential cnn{nullptr};
};

#endif// ARENAI_TRAIN_HOST_VISION_H
