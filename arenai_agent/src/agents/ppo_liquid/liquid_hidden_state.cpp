//
// Created by samuel on 06/09/2026.
//

#include "./liquid_hidden_state.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    LiquidHiddenState::LiquidHiddenState(std::shared_ptr<LiquidActor> actor)
        : actor(std::move(actor)) {}

    torch::Tensor LiquidHiddenState::get(const long batch_size) {
        if (!x.defined() || x.size(0) != batch_size)
            x = actor->initial_state(static_cast<int>(batch_size));
        return x;
    }

    void LiquidHiddenState::set(const torch::Tensor &next_x) { x = next_x.detach(); }

    void LiquidHiddenState::reset() { x = torch::Tensor(); }

}// namespace arenai::agent
