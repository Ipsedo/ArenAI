//
// Created by samuel on 23/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_CRITIC_H
#define ARENAI_TRAIN_HOST_CRITIC_H

#include <memory>

#include <torch/torch.h>

#include "vision.h"

class SacCritic final : public torch::nn::Module {
public:
    SacCritic(
        const int &nb_sensors, const int &nb_actions, const int &hidden_size_sensors,
        const int &hidden_size_actions, const int &hidden_size,
        const std::vector<std::tuple<int, int>> &vision_channels);
    torch::Tensor
    value(const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &action);

private:
    std::shared_ptr<ConvolutionNetwork> vision_encoder;
    torch::nn::Sequential sensors_encoder;
    torch::nn::Sequential action_encoder;
    torch::nn::Sequential head;
};

#endif//ARENAI_TRAIN_HOST_CRITIC_H
