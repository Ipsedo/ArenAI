//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_LIQUID_HIDDEN_STATE_H
#define ARENAI_LIQUID_HIDDEN_STATE_H

#include <memory>

#include <torch/torch.h>

#include "../../networks/recurrent/liquid_actor.h"

namespace arenai::agent {

    // Actor liquid state carried across act() calls, shared between the agent
    // (reads it before each step, advances it after) and the collector (resets
    // it when the episode ends).
    class LiquidHiddenState {
    public:
        explicit LiquidHiddenState(std::shared_ptr<LiquidActor> actor);

        // the state opening the next step; re-drawn on first use, on a batch
        // size change and after reset()
        torch::Tensor get(long batch_size);
        void set(const torch::Tensor &next_x);

        // fresh episode: the next get() re-draws every tank's initial state
        void reset();

    private:
        std::shared_ptr<LiquidActor> actor;
        torch::Tensor x;
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_HIDDEN_STATE_H
