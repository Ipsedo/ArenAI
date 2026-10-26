//
// Created by claude on 26/10/2026.
//

#ifndef ARENAI_IMPALA_VALUE_FUNCTION_H
#define ARENAI_IMPALA_VALUE_FUNCTION_H

#include <torch/torch.h>

#include "./vision.h"

namespace arenai::agent {
    // ValueFunction whose vision is encoded by an IMPALA residual CNN
    class ImpalaValueFunction final : public torch::nn::Module {
    public:
        ImpalaValueFunction(
            const int &nb_sensors, const int &hidden_size_sensors,
            const std::vector<int> &hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels);
        torch::Tensor value(const torch::Tensor &vision, const torch::Tensor &sensors);

    private:
        std::shared_ptr<ImpalaConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;
        torch::nn::Sequential head;
        torch::nn::Linear to_value;
    };
}// namespace arenai::agent

#endif//ARENAI_IMPALA_VALUE_FUNCTION_H
