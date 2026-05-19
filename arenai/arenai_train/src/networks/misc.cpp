//
// Created by samuel on 19/05/2026.
//

#include "./misc.h"

torch::Tensor Exp::forward(const torch::Tensor &x) { return torch::exp(x); }

Clamp::Clamp(const float lower_bound, const float upper_bound)
    : lower_bound(lower_bound), upper_bound(upper_bound) {}

torch::Tensor Clamp::forward(const torch::Tensor &x) {
    return torch::clamp(x, lower_bound, upper_bound);
}
