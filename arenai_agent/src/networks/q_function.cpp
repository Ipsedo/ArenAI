#include "./q_function.h"

#include "../networks_utils/init.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    QFunction::QFunction(
        const int nb_sensors, const int nb_continuous_actions, const int nb_discrete_actions,
        const int hidden_size_sensors, const std::vector<int> &hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums)
        : vision_encoder(register_module(
            "vision_encoder",
            std::make_shared<ConvolutionNetwork>(vision_channels, group_norm_nums))),
          sensors_encoder(register_module(
              "sensors_encoder",
              torch::nn::Sequential(
                  torch::nn::Linear(
                      torch::nn::LinearOptions(nb_sensors, hidden_size_sensors).bias(false)),
                  torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_sensors})),
                  torch::nn::SiLU()))),
          head(register_module("head", torch::nn::Sequential())),
          to_q(register_module(
              "to_q", torch::nn::Linear(hidden_sizes.back(), 1 << nb_discrete_actions))) {

        head->push_back(torch::nn::Linear(
            torch::nn::LinearOptions(
                hidden_size_sensors + vision_encoder->get_output_size() + nb_continuous_actions,
                hidden_sizes.front())
                .bias(false)));
        head->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_sizes.front()})));
        head->push_back(torch::nn::SiLU());

        for (size_t i = 1; i < hidden_sizes.size(); i++) {
            head->push_back(torch::nn::Linear(
                torch::nn::LinearOptions(hidden_sizes[i - 1], hidden_sizes[i]).bias(false)));
            head->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_sizes[i]})));
            head->push_back(torch::nn::SiLU());
        }

        vision_encoder->apply(init_hidden_weights);
        sensors_encoder->apply(init_hidden_weights);
        head->apply(init_hidden_weights);
        to_q->apply(init_value_output_weights);
    }

    torch::Tensor QFunction::q(
        const torch::Tensor &vision, const torch::Tensor &sensors,
        const torch::Tensor &continuous_action) {
        const auto encoded = torch::cat(
            {vision_encoder->forward(vision), sensors_encoder->forward(sensors), continuous_action},
            1);
        return to_q->forward(head->forward(encoded));
    }

    TwinQNetwork::TwinQNetwork(
        const int nb_sensors, const int nb_continuous_actions, const int nb_discrete_actions,
        const int hidden_size_sensors, const std::vector<int> &hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums)
        : q1(register_module(
            "q1", std::make_shared<QFunction>(
                      nb_sensors, nb_continuous_actions, nb_discrete_actions, hidden_size_sensors,
                      hidden_sizes, vision_channels, group_norm_nums))),
          q2(register_module(
              "q2", std::make_shared<QFunction>(
                        nb_sensors, nb_continuous_actions, nb_discrete_actions, hidden_size_sensors,
                        hidden_sizes, vision_channels, group_norm_nums))) {}

    std::pair<torch::Tensor, torch::Tensor> TwinQNetwork::q(
        const torch::Tensor &vision, const torch::Tensor &sensors,
        const torch::Tensor &continuous_action) {
        return {
            q1->q(vision, sensors, continuous_action), q2->q(vision, sensors, continuous_action)};
    }

}// namespace arenai::agent
