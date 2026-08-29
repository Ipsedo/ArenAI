//
// Created by samuel on 19/05/2026.
//

#include "./misc.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    /*
     * Exp
     */

    torch::Tensor Exp::forward(const torch::Tensor &x) { return torch::exp(x); }

    void Exp::pretty_print(std::ostream &stream) { stream << name() << "()"; }

    /*
     * Clamp
     */

    Clamp::Clamp(const float lower_bound, const float upper_bound)
        : lower_bound(lower_bound), upper_bound(upper_bound) {}

    torch::Tensor Clamp::forward(const torch::Tensor &x) {
        return torch::clamp(x, lower_bound, upper_bound);
    }

    void Clamp::pretty_print(std::ostream &stream) {
        stream << name() << "(min=" << lower_bound << ", max=" << upper_bound << ")";
    }

    /*
     * Beta distribution concentration (α or β) output layer
     */

    torch::Tensor ConcentrationOutput::forward(const torch::Tensor &input) {
        // α, β ≥ 1 keeps the Beta density unimodal
        return 1.f + torch::softplus(input);
    }

    void ConcentrationOutput::pretty_print(std::ostream &stream) { stream << name() << "()"; }

}// namespace arenai::agent
