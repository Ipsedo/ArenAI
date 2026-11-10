//
// Created by claude on 08/11/2026.
//

#ifndef ARENAI_Q_FUNCTION_H
#define ARENAI_Q_FUNCTION_H

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "./vision.h"

namespace arenai::agent {

    // Q(s, a_continuous, ·): one output per joint value of the binary discrete actions
    // (2^nb_discrete_actions), so a single forward gives the Q of every discrete choice
    class QFunction final : public torch::nn::Module {
    public:
        QFunction(
            int nb_sensors, int nb_continuous_actions, int nb_discrete_actions,
            int hidden_size_sensors, const std::vector<int> &hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums);

        // [N, 2^nb_discrete_actions]
        torch::Tensor
        q(const torch::Tensor &vision, const torch::Tensor &sensors,
          const torch::Tensor &continuous_action);

    private:
        std::shared_ptr<ConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;
        torch::nn::Sequential head;
        torch::nn::Linear to_q;
    };

    // clipped double Q: two independent critics, the target takes their minimum
    class TwinQNetwork final : public torch::nn::Module {
    public:
        TwinQNetwork(
            int nb_sensors, int nb_continuous_actions, int nb_discrete_actions,
            int hidden_size_sensors, const std::vector<int> &hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums);

        std::pair<torch::Tensor, torch::Tensor>
        q(const torch::Tensor &vision, const torch::Tensor &sensors,
          const torch::Tensor &continuous_action);

    private:
        std::shared_ptr<QFunction> q1;
        std::shared_ptr<QFunction> q2;
    };

}// namespace arenai::agent

#endif//ARENAI_Q_FUNCTION_H
