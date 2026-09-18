//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_AGENT_HOST_ENTROPY_H
#define ARENAI_AGENT_HOST_ENTROPY_H

#include <torch/torch.h>

namespace arenai::agent {

    /*
     * Lagrangian
     */

    // Inequality constraint H >= target: alpha is a Lagrange multiplier, projected on
    // [0, MAX_ALPHA] (Stooke et al. 2020) — a bonus while entropy sits under the target,
    // inactive (0) once the constraint is satisfied
    class PidLagrangianAlphaParameters final : public torch::nn::Module {
    public:
        PidLagrangianAlphaParameters(
            float k_p, float k_i, float k_d, float initial_alpha, int nb_alphas);

        void update(const torch::Tensor &entropy, const torch::Tensor &target_entropy) const;

        torch::Tensor alpha() const;

    private:
        static constexpr float MAX_ALPHA = 1.f;

        float k_p, k_i, k_d;

        torch::Tensor previous_entropy;
        torch::Tensor has_previous;

        torch::Tensor integral;
        torch::Tensor alpha_tensor;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_ENTROPY_H
