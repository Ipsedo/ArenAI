//
// Created by samuel on 22/01/2026.
//

#ifndef ARENAI_TRAIN_HOST_Q_FUNCTION_H
#define ARENAI_TRAIN_HOST_Q_FUNCTION_H

#include <vector>

#include <torch/torch.h>

#include "./vision.h"

struct q_function_response {
    torch::Tensor value;

    torch::Tensor mse_decoder;
};

class QFunction final : public torch::nn::Module {
public:
    QFunction(
        const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_actions,
        const int &hidden_size_sensors, const int &hidden_size_actions, const int &hidden_size,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums);
    q_function_response value_ohe(
        const torch::Tensor &vision, const torch::Tensor &sensors,
        const torch::Tensor &continuous_actions, const torch::Tensor &discrete_action_ohe);

    torch::Tensor value_expectation(
        const torch::Tensor &vision, const torch::Tensor &sensors,
        const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions_proba);

private:
    std::shared_ptr<VisionAutoEncoder> vision_encoder;
    torch::nn::Sequential sensors_encoder;
    torch::nn::Sequential continuous_action_encoder;
    torch::nn::Sequential discrete_action_encoder;
    torch::nn::Sequential head;
    torch::nn::Linear to_value;

    std::tuple<torch::Tensor, torch::Tensor> encode_common(
        const torch::Tensor &vision, const torch::Tensor &sensors,
        const torch::Tensor &continuous_actions);
};

#endif//ARENAI_TRAIN_HOST_Q_FUNCTION_H
