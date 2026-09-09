//
// Created by samuel on 06/09/2026.
//

#include "./liquid_critic.h"

#include "../../networks_utils/init.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    LiquidCritic::LiquidCritic(
        const int &vision_height, const int &vision_width, const int &nb_sensors,
        const int &hidden_size_sensors, const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const int &neuron_number,
        const int &unfolding_steps, const float &delta_t)
        : vision_encoder(register_module(
            "vision_encoder", std::make_shared<ConvolutionNetwork>(
                                  vision_height, vision_width, vision_channels, group_norm_nums))),
          sensors_encoder(register_module(
              "sensors_encoder",
              torch::nn::Sequential(
                  torch::nn::Linear(
                      torch::nn::LinearOptions(nb_sensors, hidden_size_sensors).bias(false)),
                  torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_sensors})),
                  torch::nn::SiLU()))),
          liquid(register_module(
              "liquid", std::make_shared<LiquidRecurrent>(
                            neuron_number, hidden_size_sensors + vision_encoder->get_output_size(),
                            neuron_number, unfolding_steps,
                            [](const torch::Tensor &t) { return torch::tanh(t); }, delta_t))),
          to_value(register_module("to_value", torch::nn::Linear(neuron_number, 1))) {

        vision_encoder->apply(init_hidden_weights);
        sensors_encoder->apply(init_hidden_weights);

        to_value->apply(init_value_output_weights);
    }

    torch::Tensor LiquidCritic::encode(const torch::Tensor &vision, const torch::Tensor &sensors) {
        return torch::cat({vision_encoder->forward(vision), sensors_encoder->forward(sensors)}, 1);
    }

    LiquidCriticOutput LiquidCritic::value(
        const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &x_t) {
        const auto [output, next_x] = liquid->forward_step(x_t, encode(vision, sensors));
        return {.value = to_value->forward(output), .next_x = next_x};
    }

    LiquidCriticOutput LiquidCritic::value_sequence(
        const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &x_t) {
        const auto batch_size = vision.size(0);
        const auto nb_steps = vision.size(1);

        // the encoders are step-wise: fold time into the row dimension
        const auto encoded =
            encode(vision.flatten(0, 1), sensors.flatten(0, 1)).reshape({batch_size, nb_steps, -1});

        auto x = x_t;

        std::vector<torch::Tensor> outputs;
        outputs.reserve(nb_steps);

        for (auto t = 0; t < nb_steps; t++) {
            auto [output, next_x] = liquid->forward_step(
                x, encoded.index({at::indexing::Slice(), t, at::indexing::Slice()}));

            x = next_x;
            outputs.push_back(output);
        }

        return {.value = to_value->forward(torch::stack(outputs, 1)), .next_x = x};
    }

    torch::Tensor LiquidCritic::initial_state(const int batch_size) const {
        return liquid->get_first_x(batch_size);
    }

}// namespace arenai::agent
