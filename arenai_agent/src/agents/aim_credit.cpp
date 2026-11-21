#include "./aim_credit.h"

#include "./ppo_objective.h"

namespace arenai::agent {

    torch::Tensor aim_potential_advantages(
        const AimPotentials &potentials, const torch::Tensor &cuts, const torch::Tensor &valids,
        const float gamma) {
        const auto rises = (gamma * potentials.after - potentials.before).unsqueeze(-1);

        return normalized(rises * (1.f - cuts), valids);
    }

}// namespace arenai::agent
