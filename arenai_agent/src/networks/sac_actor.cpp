//
// Created by claude on 08/11/2026.
//

#include "./sac_actor.h"

#include <cmath>

#include "../networks_utils/init.h"
#include "./constants.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    SacActor::SacActor(
        const int nb_sensors, const int nb_continuous_actions, const int nb_discrete_actions,
        const int hidden_size_sensors, const std::vector<int> &hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const float initial_sigma,
        const std::vector<float> &initial_discrete_probas)
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
          mu(register_module("mu", torch::nn::Linear(hidden_sizes.back(), nb_continuous_actions))),
          log_sigma(register_module(
              "log_sigma", torch::nn::Linear(hidden_sizes.back(), nb_continuous_actions))),
          discrete(register_module(
              "discrete", torch::nn::Sequential(
                              torch::nn::Linear(hidden_sizes.back(), nb_discrete_actions),
                              torch::nn::Sigmoid()))) {

        head->push_back(torch::nn::Linear(
            torch::nn::LinearOptions(
                hidden_size_sensors + vision_encoder->get_output_size(), hidden_sizes.front())
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

        // zero bias: the initial pre-squash mean is the action-range center
        mu->apply(init_mu_output_weights);

        // small weights + constant bias: the initial spread is initial_sigma whatever the state
        log_sigma->apply(init_mu_output_weights);
        torch::nn::init::constant_(log_sigma->bias, std::log(initial_sigma));

        discrete->apply([&initial_discrete_probas](Module &m) {
            init_discrete_output_weights(m, initial_discrete_probas);
        });
    }

    SacActorOutput SacActor::act(const torch::Tensor &vision, const torch::Tensor &sensors) {
        const auto encoded = head->forward(
            torch::cat({vision_encoder->forward(vision), sensors_encoder->forward(sensors)}, 1));

        return {
            .mu = mu->forward(encoded),
            .sigma = torch::exp(
                log_sigma->forward(encoded).clamp(std::log(SIGMA_MIN), std::log(SIGMA_MAX))),
            .discrete = discrete->forward(encoded)};
    }

}// namespace arenai::agent
