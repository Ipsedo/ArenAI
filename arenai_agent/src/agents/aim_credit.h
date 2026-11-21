#ifndef ARENAI_AIM_CREDIT_H
#define ARENAI_AIM_CREDIT_H

#include <torch/torch.h>

namespace arenai::agent {

    // [nb_tanks] aim potential on both sides of a step, [T, nb_tanks] once stacked
    struct AimPotentials {
        torch::Tensor before;
        torch::Tensor after;
    };

    // [T, nb_tanks, 1] normalized one-step rise of the aim potential
    torch::Tensor aim_potential_advantages(
        const AimPotentials &potentials, const torch::Tensor &cuts, const torch::Tensor &valids,
        float gamma);

}// namespace arenai::agent

#endif//ARENAI_AIM_CREDIT_H
