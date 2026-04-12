//
// Created by samuel on 23/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_ACTOR_H
#define ARENAI_TRAIN_HOST_ACTOR_H

#include <memory>

#include <torch/torch.h>

#include "./vision.h"

struct actor_response {
    torch::Tensor mu;
    torch::Tensor sigma;
    torch::Tensor discrete;
};

struct actor_response_with_encoding {
    torch::Tensor mu;
    torch::Tensor sigma;
    torch::Tensor discrete;
    torch::Tensor encoding_loss;
};

class Actor final : public torch::nn::Module {
public:
    explicit Actor(
        const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_actions,
        const int &hidden_size_sensors, const int &hidden_size,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums);

    actor_response_with_encoding
    act_with_encoding(const torch::Tensor &vision, const torch::Tensor &sensors);

    actor_response act(const torch::Tensor &vision, const torch::Tensor &sensors);

private:
    std::shared_ptr<ConvolutionNetwork> vision_encoder;
    std::shared_ptr<TransposedConvolutionNetwork> vision_decoder;
    torch::nn::Sequential sensors_encoder;

    torch::nn::Sequential head;

    torch::nn::Sequential mu;
    torch::nn::Sequential sigma;
    torch::nn::Sequential discrete;
};

#endif//ARENAI_TRAIN_HOST_ACTOR_H
