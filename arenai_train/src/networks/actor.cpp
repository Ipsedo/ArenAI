//
// Created by samuel on 23/10/2025.
//

#include "./actor.h"

#include <arenai_core/constants.h>

#include "../networks_utils/init.h"
#include "./misc.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    Actor::Actor(
        const int &vision_height, const int &vision_width, const int &nb_sensors,
        const int &nb_continuous_actions, const int &nb_discrete_actions,
        const int &hidden_size_sensors, const std::vector<int> &hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
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
          mu(register_module(
              "mu", torch::nn::Sequential(
                        torch::nn::Linear(hidden_sizes.back(), nb_continuous_actions),
                        torch::nn::Tanh()))),
          sigma(register_module(
              "sigma",
              torch::nn::Sequential(
                  torch::nn::Linear(hidden_sizes.back(), nb_continuous_actions),
                  std::make_shared<Clamp>(std::log(core::SIGMA_MIN), std::log(core::SIGMA_MAX)),
                  std::make_shared<Exp>()))),
          discrete(register_module(
              "discrete", torch::nn::Sequential(
                              torch::nn::Linear(hidden_sizes.back(), nb_discrete_actions),
                              torch::nn::Softmax(-1)))) {

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

        mu->apply(init_mu_output_weights);
        sigma->apply(init_sigma_output_weights);

        discrete->apply(init_discrete_output_weights);
    }

    ActorRawOutput Actor::act(const torch::Tensor &vision, const torch::Tensor &sensors) {
        auto vision_encoded = vision_encoder->forward(vision);
        auto sensors_encoded = sensors_encoder->forward(sensors);
        auto encoded = head->forward(torch::cat({vision_encoded, sensors_encoded}, 1));
        return {mu->forward(encoded), sigma->forward(encoded), discrete->forward(encoded)};
    }

}// namespace arenai::train
