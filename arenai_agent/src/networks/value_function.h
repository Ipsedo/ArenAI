#ifndef ARENAI_VALUE_FUNCTION_H
#define ARENAI_VALUE_FUNCTION_H

#include <torch/torch.h>

#include "./panorama.h"
#include "./vision.h"

namespace arenai::agent {
    class ValueFunction final : public torch::nn::Module {
    public:
        ValueFunction(
            const int &nb_sensors, const int &hidden_size_sensors,
            const std::vector<int> &hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums,
            const std::vector<std::tuple<int, int>> &panorama_channels,
            const std::vector<int> &panorama_group_norm_nums, const int &panorama_output_channels);
        torch::Tensor value(
            const torch::Tensor &vision, const torch::Tensor &panorama,
            const torch::Tensor &sensors);

    private:
        std::shared_ptr<ConvolutionNetwork> vision_encoder;
        std::shared_ptr<PanoramaNetwork> panorama_encoder;
        torch::nn::Sequential sensors_encoder;
        torch::nn::Sequential head;
        torch::nn::Linear to_value;
    };
}// namespace arenai::agent

#endif//ARENAI_VALUE_FUNCTION_H
