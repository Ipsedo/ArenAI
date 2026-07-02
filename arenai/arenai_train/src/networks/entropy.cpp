//
// Created by samuel on 12/10/2025.
//

#include "./entropy.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    AlphaParameter::AlphaParameter(const float initial_alpha)
        : log_alpha_tensor(
            register_parameter("log_alpha", torch::tensor({std::log(initial_alpha)}))) {}

    torch::Tensor AlphaParameter::log_alpha() { return log_alpha_tensor; }

    torch::Tensor AlphaParameter::alpha() { return log_alpha().exp(); }

}// namespace arenai::train
