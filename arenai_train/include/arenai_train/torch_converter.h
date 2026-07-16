//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_TORCH_CONVERTER_H
#define ARENAI_TRAIN_HOST_TORCH_CONVERTER_H

#include <tuple>

#include <torch/torch.h>

#include <arenai_core/types.h>

namespace arenai::train {

    std::vector<core::Action> tensor_to_actions(
        const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions);

    std::tuple<torch::Tensor, torch::Tensor>
    states_to_tensor(const std::vector<core::State> &states, int vision_height, int vision_width);
    std::tuple<torch::Tensor, torch::Tensor>
    state_to_tensor(const core::State &state, int vision_height, int vision_width);

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_TORCH_CONVERTER_H
