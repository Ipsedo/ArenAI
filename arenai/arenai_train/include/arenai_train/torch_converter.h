//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_TORCH_CONVERTER_H
#define ARENAI_TRAIN_HOST_TORCH_CONVERTER_H

#include <tuple>

#include <torch/torch.h>

#include <arenai_core/types.h>

std::vector<Action>
tensor_to_actions(const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions);

std::tuple<torch::Tensor, torch::Tensor>
states_to_tensor(const std::vector<State> &states, int vision_height, int vision_width);
std::tuple<torch::Tensor, torch::Tensor>
state_to_tensor(const State &state, int vision_height, int vision_width);

#endif//ARENAI_TRAIN_HOST_TORCH_CONVERTER_H
