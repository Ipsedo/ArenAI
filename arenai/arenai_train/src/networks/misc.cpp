//
// Created by samuel on 19/05/2026.
//

#include "./misc.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    /*
     * Exp
     */

    torch::Tensor Exp::forward(const torch::Tensor &x) { return torch::exp(x); }

    void Exp::pretty_print(std::ostream &stream) const { stream << name() << "()"; }

    /*
     * Clamp
     */

    Clamp::Clamp(const float lower_bound, const float upper_bound)
        : lower_bound(lower_bound), upper_bound(upper_bound) {}

    torch::Tensor Clamp::forward(const torch::Tensor &x) {
        return torch::clamp(x, lower_bound, upper_bound);
    }

    void Clamp::pretty_print(std::ostream &stream) const {
        stream << name() << "(min=" << lower_bound << ", max=" << upper_bound << ")";
    }

}// namespace arenai::train
