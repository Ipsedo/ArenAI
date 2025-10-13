//
// Created by samuel on 02/10/2025.
//

#ifndef PHYVR_AGENT_H
#define PHYVR_AGENT_H

#include <memory>
#include <tuple>

#include <torch/torch.h>

#include "vision.h"

struct actor_response {
    torch::Tensor mu;
    torch::Tensor sigma;
};

class SacActor final : public torch::nn::Module {
public:
    explicit SacActor(
        const int &nb_sensors, const int &nb_actions, const int &hidden_size_sensors,
        const int &hidden_size);
    actor_response act(const torch::Tensor &vision, const torch::Tensor &sensors);

private:
    std::shared_ptr<ConvolutionNetwork> vision_encoder;
    torch::nn::Sequential sensors_encoder;

    torch::nn::Sequential head;
    torch::nn::Sequential mu;
    torch::nn::Sequential sigma;
};

class SacCritic final : public torch::nn::Module {
public:
    SacCritic(
        const int &nb_sensors, const int &nb_actions, const int &hidden_size_sensors,
        const int &hidden_size_actions, const int &hidden_size);
    torch::Tensor
    value(const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &action);

private:
    std::shared_ptr<ConvolutionNetwork> vision_encoder;
    torch::nn::Sequential sensors_encoder;
    torch::nn::Sequential action_encoder;
    torch::nn::Sequential head;
};

#endif// PHYVR_AGENT_H
