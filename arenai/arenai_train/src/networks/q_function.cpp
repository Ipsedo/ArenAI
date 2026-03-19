//
// Created by samuel on 22/01/2026.
//

#include "./q_function.h"

#include "./init.h"

QFunction::QFunction(
    const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_actions,
    const int &hidden_size_sensors, const int &hidden_size_actions, const int &hidden_size,
    const std::vector<std::tuple<int, int>> &vision_channels,
    const std::vector<int> &group_norm_nums)
    : vision_encoder(register_module(
        "vision_encoder", std::make_shared<ConvolutionNetwork>(vision_channels, group_norm_nums))),
      sensors_encoder(register_module(
          "sensors_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_sensors, hidden_size_sensors),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_sensors})),
              torch::nn::SiLU()))),
      continuous_action_encoder(register_module(
          "continuous_action_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_continuous_actions, hidden_size_actions),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_actions})),
              torch::nn::SiLU()))),
      discrete_action_encoder(register_module(
          "discrete_action_encoder",
          torch::nn::Sequential(
              torch::nn::Linear(nb_discrete_actions, hidden_size_actions),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_actions})),
              torch::nn::SiLU()))),
      head(register_module(
          "head",
          torch::nn::Sequential(
              torch::nn::Linear(
                  2 * hidden_size_actions + hidden_size_sensors + vision_encoder->get_output_size(),
                  hidden_size),
              torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})), torch::nn::SiLU(),
              torch::nn::Linear(hidden_size, 1)))) {
    apply(init_weights);
}

torch::Tensor QFunction::value_ohe(
    const torch::Tensor &vision, const torch::Tensor &sensors,
    const torch::Tensor &continuous_actions, const torch::Tensor &discrete_action_ohe) {
    const auto common_encoded = encode_common(vision, sensors, continuous_actions);
    const auto discrete_action_encoded = discrete_action_encoder->forward(discrete_action_ohe);

    const auto encoded_hidden = torch::cat({common_encoded, discrete_action_encoded}, 1);

    return head->forward(encoded_hidden);
}

torch::Tensor QFunction::value_expectation(
    const torch::Tensor &vision, const torch::Tensor &sensors,
    const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions_proba) {
    const auto batch_size = vision.size(0);
    const auto nb_discrete_actions = discrete_actions_proba.size(1);

    const auto common_encoded = encode_common(vision, sensors, continuous_actions);

    torch::Tensor expected_value = torch::zeros({batch_size, 1}, discrete_actions_proba.options());

    for (int k = 0; k < nb_discrete_actions; k++) {
        const auto one_hot =
            torch::zeros({batch_size, nb_discrete_actions}, discrete_actions_proba.options())
                .index_put_({torch::indexing::Slice(), k}, 1.0);

        const auto encoded_discrete_actions = discrete_action_encoder->forward(one_hot);

        const auto value = head->forward(torch::cat({common_encoded, encoded_discrete_actions}, 1));
        const auto proba = discrete_actions_proba.index({torch::indexing::Slice(), k}).unsqueeze(1);

        expected_value += proba * value;
    }

    return expected_value;
}

torch::Tensor QFunction::encode_common(
    const torch::Tensor &vision, const torch::Tensor &sensors,
    const torch::Tensor &continuous_actions) {
    const auto vision_encoded = vision_encoder->forward(vision);
    const auto sensors_encoded = sensors_encoder->forward(sensors);
    const auto action_encoded = continuous_action_encoder->forward(continuous_actions);

    return torch::cat({vision_encoded, sensors_encoded, action_encoded}, 1);
}
