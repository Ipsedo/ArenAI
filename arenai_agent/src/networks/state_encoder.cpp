#include "./state_encoder.h"

#include "../networks_utils/init.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    StateEncoder::StateEncoder(
        const int nb_sensors, const int hidden_size_sensors,
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
          output_size(vision_encoder->get_output_size() + hidden_size_sensors) {
        vision_encoder->apply(init_hidden_weights);
        sensors_encoder->apply(init_hidden_weights);
    }

    torch::Tensor StateEncoder::forward(const torch::Tensor &vision, const torch::Tensor &sensors) {
        return torch::cat({vision_encoder->forward(vision), sensors_encoder->forward(sensors)}, 1);
    }

    int StateEncoder::get_output_size() const { return output_size; }

}// namespace arenai::agent
