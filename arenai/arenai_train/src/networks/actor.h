//
// Created by samuel on 23/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_ACTOR_H
#define ARENAI_TRAIN_HOST_ACTOR_H

#include <memory>

#include <torch/torch.h>

#include "vision.h"

struct actor_response {
    torch::Tensor mu;
    torch::Tensor sigma;
};

class Actor final : public torch::nn::Module {
public:
    explicit Actor(
        const int &nb_sensors, const int &nb_actions, const int &hidden_size_sensors,
        const int &hidden_size, const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums);
    actor_response act(const torch::Tensor &vision, const torch::Tensor &sensors);

private:
    std::shared_ptr<ConvolutionNetwork> vision_encoder;
    torch::nn::Sequential sensors_encoder;

    torch::nn::Sequential head;
    torch::nn::Sequential mu;
    torch::nn::Sequential sigma;
};

#endif//ARENAI_TRAIN_HOST_ACTOR_H
