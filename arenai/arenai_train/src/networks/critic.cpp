//
// Created by samuel on 23/10/2025.
//

#include "./critic.h"

#include "./init.h"

Critic::Critic(
    const int &nb_sensors, const int &hidden_size_sensors, const int &hidden_size,
    const std::vector<std::tuple<int, int>> &vision_channels,
    const std::vector<int> &group_norm_nums)
    : vision_encoder(register_module(
        "vision_encoder", std::make_shared<ConvolutionNetwork>(vision_channels, group_norm_nums))),
      sensors_encoder(register_module(
          "sensors_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_sensors, hidden_size_sensors), torch::nn::SiLU()))),
      head(register_module(
          "head", torch::nn::Sequential(
                      torch::nn::Linear(
                          hidden_size_sensors + vision_encoder->get_output_size(), hidden_size),
                      torch::nn::Linear(hidden_size, 1)))) {
    apply(init_weights);
}

torch::Tensor Critic::value(const torch::Tensor &vision, const torch::Tensor &sensors) {
    const auto encoded_state =
        torch::cat({vision_encoder->forward(vision), sensors_encoder->forward(sensors)}, 1);
    return head->forward(encoded_state);
}
