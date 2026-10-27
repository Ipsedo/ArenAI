//
// Created by claude on 27/10/2026.
//

#include "./liquid_impala_hidden_state.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    LiquidImpalaHiddenState::LiquidImpalaHiddenState(std::shared_ptr<LiquidImpalaActor> actor)
        : actor(std::move(actor)) {}

    torch::Tensor LiquidImpalaHiddenState::get(const long batch_size) {
        if (!x.defined() || x.size(0) != batch_size)
            x = actor->initial_state(static_cast<int>(batch_size));
        return x;
    }

    void LiquidImpalaHiddenState::set(const torch::Tensor &next_x) { x = next_x.detach(); }

    void LiquidImpalaHiddenState::reset() { x = torch::Tensor(); }

}// namespace arenai::agent
