//
// Created by samuel on 23/10/2025.
//

#include "./actor.h"

#include "./init.h"

SacActor::SacActor(
    const int &nb_sensors, const int &nb_actions, const int &hidden_size_sensors,
    const int &hidden_size, const std::vector<std::tuple<int, int>> &vision_channels,
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
                      torch::nn::SiLU()))),
      mu(register_module(
          "mu",
          torch::nn::Sequential(torch::nn::Linear(hidden_size, nb_actions), torch::nn::Tanh()))),
      sigma(register_module(
          "sigma", torch::nn::Sequential(
                       torch::nn::Linear(hidden_size, nb_actions), torch::nn::Softplus()))) {
    apply(init_weights);
}

actor_response SacActor::act(const torch::Tensor &vision, const torch::Tensor &sensors) {
    auto vision_encoded = vision_encoder->forward(vision);
    auto sensors_encoded = sensors_encoder->forward(sensors);
    auto encoded = head->forward(torch::cat({vision_encoded, sensors_encoded}, 1));
    return {mu->forward(encoded), sigma->forward(encoded)};
}
