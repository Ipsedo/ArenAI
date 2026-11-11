#include "./q_function.h"

#include "../networks_utils/init.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    QFunction::QFunction(
        const int encoding_size, const int nb_continuous_actions, const int nb_discrete_actions,
        const std::vector<int> &hidden_sizes)
        : head(register_module("head", torch::nn::Sequential())),
          to_q(register_module(
              "to_q", torch::nn::Linear(hidden_sizes.back(), 1 << nb_discrete_actions))) {

        head->push_back(torch::nn::Linear(
            torch::nn::LinearOptions(encoding_size + nb_continuous_actions, hidden_sizes.front())
                .bias(false)));
        head->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_sizes.front()})));
        head->push_back(torch::nn::SiLU());

        for (size_t i = 1; i < hidden_sizes.size(); i++) {
            head->push_back(torch::nn::Linear(
                torch::nn::LinearOptions(hidden_sizes[i - 1], hidden_sizes[i]).bias(false)));
            head->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_sizes[i]})));
            head->push_back(torch::nn::SiLU());
        }

        head->apply(init_hidden_weights);
        to_q->apply(init_value_output_weights);
    }

    torch::Tensor
    QFunction::q(const torch::Tensor &encoding, const torch::Tensor &continuous_action) {
        return to_q->forward(head->forward(torch::cat({encoding, continuous_action}, 1)));
    }

    TwinQNetwork::TwinQNetwork(
        const std::shared_ptr<StateEncoder> &encoder, const int nb_continuous_actions,
        const int nb_discrete_actions, const std::vector<int> &hidden_sizes)
        : encoder(register_module("encoder", encoder)),
          q1(register_module(
              "q1", std::make_shared<QFunction>(
                        encoder->get_output_size(), nb_continuous_actions, nb_discrete_actions,
                        hidden_sizes))),
          q2(register_module(
              "q2", std::make_shared<QFunction>(
                        encoder->get_output_size(), nb_continuous_actions, nb_discrete_actions,
                        hidden_sizes))) {}

    torch::Tensor TwinQNetwork::encode(const torch::Tensor &vision, const torch::Tensor &sensors) {
        return encoder->forward(vision, sensors);
    }

    std::pair<torch::Tensor, torch::Tensor> TwinQNetwork::q_from_encoding(
        const torch::Tensor &encoding, const torch::Tensor &continuous_action) {
        return {q1->q(encoding, continuous_action), q2->q(encoding, continuous_action)};
    }

    std::pair<torch::Tensor, torch::Tensor> TwinQNetwork::q(
        const torch::Tensor &vision, const torch::Tensor &sensors,
        const torch::Tensor &continuous_action) {
        return q_from_encoding(encode(vision, sensors), continuous_action);
    }

}// namespace arenai::agent
