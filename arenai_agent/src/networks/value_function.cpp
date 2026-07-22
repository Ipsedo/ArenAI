//
// Created by samuel on 22/07/2026.
//

#include "./value_function.h"

#include "../networks_utils/init.h"

using namespace arenai;
using namespace arenai::agent;

ValueFunction::ValueFunction(
    const int &vision_height, const int &vision_width, const int &nb_sensors,
    const int &hidden_size_sensors, const int &hidden_size_actions,
    const std::vector<int> &hidden_sizes, const std::vector<std::tuple<int, int>> &vision_channels,
    const std::vector<int> &group_norm_nums)
    : vision_encoder(register_module(
        "vision_encoder", std::make_shared<ConvolutionNetwork>(
                              vision_height, vision_width, vision_channels, group_norm_nums))),
      sensors_encoder(register_module(
          "sensors_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_sensors, hidden_size_sensors),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_sensors})),
              torch::nn::GELU()))),
      head(register_module("head", torch::nn::Sequential())),
      to_value(register_module("to_value", torch::nn::Linear(hidden_sizes.back(), 1))) {

    head->push_back(torch::nn::Linear(
        hidden_size_sensors + vision_encoder->get_output_size(), hidden_sizes.front()));
    head->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_sizes.front()})));
    head->push_back(torch::nn::GELU());

    for (int i = 1; i < hidden_sizes.size(); i++) {
        const auto curr_size = hidden_sizes[i - 1];
        const auto next_size = hidden_sizes[i];
        head->push_back(torch::nn::Linear(curr_size, next_size));
        head->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({next_size})));
        head->push_back(torch::nn::GELU());
    }

    vision_encoder->apply(init_hidden_weights);
    sensors_encoder->apply(init_hidden_weights);
    head->apply(init_hidden_weights);

    to_value->apply(init_value_output_weights);
}

torch::Tensor ValueFunction::value(const torch::Tensor &vision, const torch::Tensor &sensors) {
    const auto vision_encoded = vision_encoder->forward(vision);
    const auto sensors_encoded = sensors_encoder->forward(sensors);

    const auto encoded = torch::cat({vision_encoded, sensors_encoded}, 1);
    const auto hidden = head->forward(encoded);

    return to_value->forward(hidden);
}
