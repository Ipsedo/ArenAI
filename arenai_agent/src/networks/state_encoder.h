#ifndef ARENAI_STATE_ENCODER_H
#define ARENAI_STATE_ENCODER_H

#include <memory>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "./vision.h"

namespace arenai::agent {

    class StateEncoder final : public torch::nn::Module {
    public:
        StateEncoder(
            int nb_sensors, int hidden_size_sensors,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums);

        // [N, vision output size + hidden_size_sensors]
        torch::Tensor forward(const torch::Tensor &vision, const torch::Tensor &sensors);

        int get_output_size() const;

    private:
        std::shared_ptr<ConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;
        int output_size;
    };

}// namespace arenai::agent

#endif//ARENAI_STATE_ENCODER_H
