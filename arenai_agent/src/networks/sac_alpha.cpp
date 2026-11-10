//
// Created by claude on 10/11/2026.
//

#include "./sac_alpha.h"

#include <cmath>

namespace arenai::agent {

    SacAlpha::SacAlpha(const float initial_alpha, const int nb_alphas)
        : log_alpha(
            register_parameter("log_alpha", torch::full({nb_alphas}, std::log(initial_alpha)))) {}

    torch::Tensor SacAlpha::alpha() const { return torch::exp(log_alpha); }

}// namespace arenai::agent
