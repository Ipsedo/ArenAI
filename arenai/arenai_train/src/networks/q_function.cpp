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
      vision_decoder(register_module(
          "vision_decoder",
          std::make_shared<TransposedConvolutionNetwork>(
              vision_channels, group_norm_nums, vision_encoder->get_output_height(),
              vision_encoder->get_output_width()))),
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

q_function_response_with_encode QFunction::value_ohe(
    const torch::Tensor &vision, const torch::Tensor &sensors,
    const torch::Tensor &continuous_actions, const torch::Tensor &discrete_action_ohe) {

    const auto encoded_image = vision_encoder->forward(vision);
    const auto common_encoded = encode_and_cat_common(encoded_image, sensors, continuous_actions);

    const auto discrete_action_encoded = discrete_action_encoder->forward(discrete_action_ohe);

    const auto encoded_hidden = torch::cat({common_encoded, discrete_action_encoded}, 1);

    const auto loss_encoding = vision_decoder->forward_to_loss(encoded_image, vision);

    return {head->forward(encoded_hidden), loss_encoding};
}

q_function_response QFunction::value_expectation(
    const torch::Tensor &vision, const torch::Tensor &sensors,
    const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions_proba) {
    const auto batch_size = vision.size(0);
    const auto nb_discrete_actions = discrete_actions_proba.size(1);

    const auto encoded_image = vision_encoder->forward(vision);
    const auto common_encoded = encode_and_cat_common(encoded_image, sensors, continuous_actions);

    const auto one_hots = torch::eye(nb_discrete_actions, discrete_actions_proba.options());

    auto result = torch::zeros({batch_size, 1}, common_encoded.options().requires_grad(false));
    for (int a = 0; a < nb_discrete_actions; a++) {
        const auto discrete_encoded =
            discrete_action_encoder->forward(one_hots[a].unsqueeze(0).expand({batch_size, -1}));

        const auto q_a = head->forward(torch::cat({common_encoded, discrete_encoded}, 1));

        result = result + discrete_actions_proba.select(1, a).unsqueeze(1) * q_a;
    }

    return {result};
}

torch::Tensor QFunction::encode_and_cat_common(
    const torch::Tensor &encoded_vision, const torch::Tensor &sensors,
    const torch::Tensor &continuous_actions) {
    const auto sensors_encoded = sensors_encoder->forward(sensors);
    const auto action_encoded = continuous_action_encoder->forward(continuous_actions);

    return torch::cat({encoded_vision.flatten(1, -1), sensors_encoded, action_encoded}, 1);
}
