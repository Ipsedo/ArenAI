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

private:
    torch::nn::Sequential cnn{nullptr};
    int output_size;
};

class TransposedConvolutionNetwork final : public torch::nn::Module {
public:
    explicit TransposedConvolutionNetwork(
        const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums);

    torch::Tensor forward(const torch::Tensor &input);

private:
    torch::nn::Sequential tr_cnn{nullptr};
};

struct vision_output {
    torch::Tensor encoded_image;
    torch::Tensor decoded_image_diff;
};

class VisionAutoEncoder final : public torch::nn::Module {
public:
    explicit VisionAutoEncoder(
        const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums);

    vision_output forward(const torch::Tensor &input) const;

    int get_output_size() const;

private:
    std::shared_ptr<ConvolutionNetwork> encoder;
    std::shared_ptr<TransposedConvolutionNetwork> decoder;
};

#endif// ARENAI_TRAIN_HOST_VISION_H
