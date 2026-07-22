//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_AGENT_HOST_TORCH_CONVERTER_H
#define ARENAI_AGENT_HOST_TORCH_CONVERTER_H

#include <tuple>

#include <torch/torch.h>

#include <arenai_core/types.h>

#include "../agents/torch_types.h"

namespace arenai::agent {

    std::vector<core::Action> tensor_to_actions(
        const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions);

    TorchState
    states_to_tensor(const std::vector<core::State> &states, int vision_height, int vision_width);
    TorchState state_to_tensor(const core::State &state, int vision_height, int vision_width);

    TorchStep steps_to_tensor(
        const std::vector<std::tuple<core::State, core::Reward, core::IsDone, core::IsTruncated>>
            &steps,
        int vision_height, int vision_width);

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_TORCH_CONVERTER_H
