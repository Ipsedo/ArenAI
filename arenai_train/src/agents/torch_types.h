//
// Created by samuel on 21/07/2026.
//

#ifndef ARENAI_TORCH_TYPES_H
#define ARENAI_TORCH_TYPES_H

#include <torch/torch.h>

namespace arenai::train {
    struct TorchState {
        torch::Tensor vision;
        torch::Tensor proprioception;
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
}// namespace arenai::train

#endif//ARENAI_TORCH_TYPES_H
