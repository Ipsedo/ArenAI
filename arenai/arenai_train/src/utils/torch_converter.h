//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_TORCH_CONVERTER_H
#define ARENAI_TRAIN_HOST_TORCH_CONVERTER_H

#include <tuple>

#include <torch/torch.h>

#include <arenai_core/types.h>

std::vector<Action> tensor_to_actions(const torch::Tensor &actions_tensor);

std::tuple<torch::Tensor, torch::Tensor> states_to_tensor(const std::vector<State> &states);

#endif//ARENAI_TRAIN_HOST_TORCH_CONVERTER_H
