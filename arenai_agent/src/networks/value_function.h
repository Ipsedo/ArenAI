//
// Created by samuel on 22/07/2026.
//

#ifndef ARENAI_VALUE_FUNCTION_H
#define ARENAI_VALUE_FUNCTION_H

#include <torch/torch.h>

#include "./vision.h"

namespace arenai::agent {
    class ValueFunction final : public torch::nn::Module {
    public:
        ValueFunction(
            const int &vision_height, const int &vision_width, const int &nb_sensors,
            const int &hidden_size_sensors, const std::vector<int> &hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums);
        torch::Tensor value(const torch::Tensor &vision, const torch::Tensor &sensors);

    private:
        std::shared_ptr<ConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;
        torch::nn::Sequential head;
        torch::nn::Linear to_value;
    };
}// namespace arenai::agent

#endif//ARENAI_VALUE_FUNCTION_H
