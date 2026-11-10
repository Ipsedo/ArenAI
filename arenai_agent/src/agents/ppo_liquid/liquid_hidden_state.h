#ifndef ARENAI_LIQUID_HIDDEN_STATE_H
#define ARENAI_LIQUID_HIDDEN_STATE_H

#include <memory>

#include <torch/torch.h>

#include "../../networks/recurrent/liquid_actor.h"

namespace arenai::agent {

    class LiquidHiddenState {
    public:
        explicit LiquidHiddenState(std::shared_ptr<LiquidActor> actor);

        torch::Tensor get(long batch_size);
        void set(const torch::Tensor &next_x);

        void reset();

    private:
        std::shared_ptr<LiquidActor> actor;
        torch::Tensor x;
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_HIDDEN_STATE_H
