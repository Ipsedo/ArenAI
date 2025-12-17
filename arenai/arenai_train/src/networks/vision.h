//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_VISION_H
#define ARENAI_TRAIN_HOST_VISION_H

#include <torch/torch.h>

class ConvolutionNetwork final : public torch::nn::Module {
public:
    explicit ConvolutionNetwork(
        const std::vector<std::tuple<int, int>> &channels, int num_group_norm);

    torch::Tensor forward(const torch::Tensor &input);

    int get_output_size() const;

private:
    torch::nn::Sequential cnn{nullptr};
    int output_size;
};

#endif// ARENAI_TRAIN_HOST_VISION_H
