//
// Created by samuel on 03/10/2025.
//

#include "./agent.h"

SacActor::SacActor(
    const int &nb_sensors, const int &nb_actions, const int &hidden_size_sensors,
    const int &hidden_size)
    : vision_encoder(register_module("vision_encoder", std::make_shared<ConvolutionNetwork>())),
      sensors_encoder(register_module(
          "sensors_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_sensors, hidden_size_sensors), torch::nn::SiLU(),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_sensors}))))),
      head(register_module(
          "head",
          torch::nn::Sequential(
              torch::nn::Linear(hidden_size_sensors + 1 * 1 * 128, hidden_size), torch::nn::SiLU(),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}))))),
      mu(register_module(
          "mu",
          torch::nn::Sequential(torch::nn::Linear(hidden_size, nb_actions), torch::nn::Tanh()))),
      sigma(register_module(
          "sigma", torch::nn::Sequential(
                       torch::nn::Linear(hidden_size, nb_actions), torch::nn::Softplus()))) {}

actor_response SacActor::act(const torch::Tensor &vision, const torch::Tensor &sensors) {
    auto vision_encoded = vision_encoder->forward(vision);
    auto sensors_encoded = sensors_encoder->forward(sensors);
    auto encoded = head->forward(torch::cat({vision_encoded, sensors_encoded}, 1));
    return {mu->forward(encoded), sigma->forward(encoded)};
}

SacCritic::SacCritic(
    const int &nb_sensors, const int &nb_actions, const int &hidden_size_sensors,
    const int &hidden_size_actions, const int &hidden_size)
    : vision_encoder(register_module("vision_encoder", std::make_shared<ConvolutionNetwork>())),
      sensors_encoder(register_module(
          "sensors_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_sensors, hidden_size_sensors), torch::nn::SiLU(),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_sensors}))))),
      action_encoder(register_module(
          "action_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_actions, hidden_size_actions), torch::nn::SiLU(),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_actions}))))),
      head(register_module(
          "head",
          torch::nn::Sequential(
              torch::nn::Linear(
                  hidden_size_actions + hidden_size_sensors + 1 * 1 * 128, hidden_size),
              torch::nn::SiLU(), torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})),
              torch::nn::Linear(hidden_size, 1)))) {}

torch::Tensor SacCritic::value(
    const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &action) {
    auto vision_encoded = vision_encoder->forward(vision);
    auto sensors_encoded = sensors_encoder->forward(sensors);
    auto action_encoded = action_encoder->forward(action);

    return head->forward(torch::cat({vision_encoded, sensors_encoded, action_encoded}, 1));
}
