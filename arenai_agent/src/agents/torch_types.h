//
// Created by samuel on 21/07/2026.
//

#ifndef ARENAI_TORCH_TYPES_H
#define ARENAI_TORCH_TYPES_H

#include <torch/torch.h>

namespace arenai::agent {
    struct TorchState {
        torch::Tensor vision;
        torch::Tensor proprioception;
        // [N, 1] float, 1 when a fire input would shoot, 0 when it has no effect
        torch::Tensor fire_mask;
    };

    struct TorchAction {
        torch::Tensor continuous_action;
        torch::Tensor discrete_action;
    };

    struct TorchStep {
        TorchState states;
        torch::Tensor rewards;
        torch::Tensor is_done;
        torch::Tensor is_truncated;
    };
}// namespace arenai::agent

#endif//ARENAI_TORCH_TYPES_H
