//
// Created by samuel on 22/01/2026.
//

#include "./q_function.h"

#include "./init.h"

QFunction::QFunction(
    const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_actions,
    const int &hidden_size_sensors, const int &hidden_size_actions,
    const std::vector<int> &hidden_sizes, const std::vector<std::tuple<int, int>> &vision_channels,
    const std::vector<int> &group_norm_nums)
    : vision_encoder(register_module(
        "vision_encoder", std::make_shared<ConvolutionNetwork>(vision_channels, group_norm_nums))),
      sensors_encoder(register_module(
          "sensors_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_sensors, hidden_size_sensors),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_sensors})),
              torch::nn::GELU()))),
      continuous_action_encoder(register_module(
          "continuous_action_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_continuous_actions, hidden_size_actions),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_actions})),
              torch::nn::GELU()))),
      discrete_action_encoder(register_module(
          "discrete_action_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_discrete_actions, hidden_size_actions),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_actions})),
              torch::nn::GELU()))),
      head(register_module("head", torch::nn::Sequential())),
      to_value(register_module("to_value", torch::nn::Linear(hidden_sizes.back(), 1))) {

    head->push_back(torch::nn::Linear(
        2 * hidden_size_actions + hidden_size_sensors + vision_encoder->get_output_size(),
        hidden_sizes.front()));
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
    continuous_action_encoder->apply(init_hidden_weights);
    discrete_action_encoder->apply(init_hidden_weights);
    head->apply(init_hidden_weights);

    to_value->apply(init_value_output_weights);
}

torch::Tensor QFunction::value_ohe(
    const torch::Tensor &vision, const torch::Tensor &sensors,
    const torch::Tensor &continuous_actions, const torch::Tensor &discrete_action_ohe) {
    const auto common_encoded = encode_common(vision, sensors, continuous_actions);
    const auto discrete_action_encoded = discrete_action_encoder->forward(discrete_action_ohe);

    const auto encoded_hidden = torch::cat({common_encoded, discrete_action_encoded}, 1);

    return to_value->forward(head->forward(encoded_hidden));
}

torch::Tensor QFunction::value_expectation(
    const torch::Tensor &vision, const torch::Tensor &sensors,
    const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions_proba) {
    const auto batch_size = vision.size(0);
    const auto nb_discrete_actions = discrete_actions_proba.size(1);

    const auto common_encoded = encode_common(vision, sensors, continuous_actions);
    const auto one_hots = torch::eye(nb_discrete_actions, discrete_actions_proba.options());

    auto result = torch::zeros({batch_size, 1}, common_encoded.options());

    for (int a = 0; a < nb_discrete_actions; a++) {
        const auto discrete_encoded =
            discrete_action_encoder->forward(one_hots[a].unsqueeze(0).expand({batch_size, -1}));

        const auto q_a =
            to_value->forward(head->forward(torch::cat({common_encoded, discrete_encoded}, 1)));

        result = result + discrete_actions_proba.select(1, a).unsqueeze(1) * q_a;
    }

    return result;
}

torch::Tensor QFunction::encode_common(
    const torch::Tensor &vision, const torch::Tensor &sensors,
    const torch::Tensor &continuous_actions) {
    const auto vision_encoded = vision_encoder->forward(vision);
    const auto sensors_encoded = sensors_encoder->forward(sensors);
    const auto action_encoded = continuous_action_encoder->forward(continuous_actions);

    return torch::cat({vision_encoded, sensors_encoded, action_encoded}, 1);
}
