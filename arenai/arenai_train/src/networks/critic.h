//
// Created by samuel on 23/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_CRITIC_H
#define ARENAI_TRAIN_HOST_CRITIC_H

#include <memory>

#include <torch/torch.h>

#include "./vision.h"

class Critic final : public torch::nn::Module {
public:
    Critic(
        const int &nb_sensors, const int &hidden_size_sensors, const int &hidden_size,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums);
    torch::Tensor value(const torch::Tensor &vision, const torch::Tensor &sensors);

private:
    std::shared_ptr<ConvolutionNetwork> vision_encoder;
    torch::nn::Sequential sensors_encoder;
    torch::nn::Sequential head;
};

#endif//ARENAI_TRAIN_HOST_CRITIC_H
