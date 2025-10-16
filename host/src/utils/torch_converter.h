//
// Created by samuel on 12/10/2025.
//

#ifndef PHYVR_TRAIN_HOST_TORCH_CONVERTER_H
#define PHYVR_TRAIN_HOST_TORCH_CONVERTER_H

#include <tuple>

#include <torch/torch.h>

#include <phyvr_core/types.h>

std::vector<Action> tensor_to_actions(const torch::Tensor &actions_tensor);

std::tuple<torch::Tensor, torch::Tensor> states_to_tensor(const std::vector<State> &states);
std::tuple<torch::Tensor, torch::Tensor> state_to_tensor(const State &state);

#endif//PHYVR_TRAIN_HOST_TORCH_CONVERTER_H
