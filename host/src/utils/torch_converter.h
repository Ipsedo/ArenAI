//
// Created by samuel on 12/10/2025.
//

#ifndef PHYVR_TRAIN_HOST_TORCH_CONVERTER_H
#define PHYVR_TRAIN_HOST_TORCH_CONVERTER_H

#include <tuple>

#include <torch/torch.h>

#include <phyvr_core/types.h>

std::vector<Action> actions_tensor_to_core(const torch::Tensor &actions_tensor);

std::tuple<torch::Tensor, torch::Tensor> state_core_to_tensor(const std::vector<State> &states);

#endif//PHYVR_TRAIN_HOST_TORCH_CONVERTER_H
