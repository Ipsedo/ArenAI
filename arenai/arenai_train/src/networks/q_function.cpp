//
// Created by samuel on 22/01/2026.
//

#include "./q_function.h"

#include "./init.h"

QFunction::QFunction(
    const int &nb_sensors, const int &nb_actions, const int &hidden_size_sensors,
    const int &hidden_size_actions, const int &hidden_size,
    const std::vector<std::tuple<int, int>> &vision_channels,
    const std::vector<int> &group_norm_nums)
    : vision_encoder(register_module(
        "vision_encoder", std::make_shared<ConvolutionNetwork>(vision_channels, group_norm_nums))),
      sensors_encoder(register_module(
          "sensors_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_sensors, hidden_size_sensors), torch::nn::SiLU()))),
      action_encoder(register_module(
          "action_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_actions, hidden_size_actions), torch::nn::SiLU()))),
      head(register_module(
          "head",
          torch::nn::Sequential(
              torch::nn::Linear(
                  hidden_size_actions + hidden_size_sensors + vision_encoder->get_output_size(),
                  hidden_size),
              torch::nn::Linear(hidden_size, 1)))) {
    apply(init_weights);
}

torch::Tensor QFunction::value(
    const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &action) {
    auto vision_encoded = vision_encoder->forward(vision);
    auto sensors_encoded = sensors_encoder->forward(sensors);
    auto action_encoded = action_encoder->forward(action);

    return head->forward(torch::cat({vision_encoded, sensors_encoded, action_encoded}, 1));
}
