//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_AGENT_HOST_ENTROPY_H
#define ARENAI_AGENT_HOST_ENTROPY_H

#include <torch/torch.h>

namespace arenai::agent {

    /*
     * Base class
     */

    class AlphaParameters : public torch::nn::Module {
    public:
        explicit AlphaParameters(float initial_alpha, int nb_alphas);

        virtual torch::Tensor log_alpha();
        torch::Tensor alpha();

    private:
        torch::Tensor log_alpha_tensor;
    };

    class ClampedAlphaParameters final : public AlphaParameters {
    public:
        explicit ClampedAlphaParameters(
            float initial_alpha, float min_alpha, float max_alpha, int nb_alphas);

        torch::Tensor log_alpha() override;

    private:
        float min_log_alpha;
        float max_log_alpha;
    };

    class AbstractTargetEntropy : public torch::nn::Module {
    public:
        virtual torch::Tensor target_entropy() = 0;
    };

    /*
     * Constants
     */

    class ConstantTargetEntropy : public AbstractTargetEntropy {
    public:
        explicit ConstantTargetEntropy(float initial_target);

        torch::Tensor target_entropy() override;

    private:
        torch::Tensor initial_target;
    };

    /*
     * Warmup
     */

    class CosineAnnealingTargetEntropy : public AbstractTargetEntropy {
    public:
        CosineAnnealingTargetEntropy(float initial_value, float final_value, int warmup_step);
        torch::Tensor target_entropy() override;

    private:
        float initial;
        float final;
        int warmup_step;

        torch::Tensor current_step;
    };

    /*
     * Lagrangian
     */

    class PidLagrangianAlphaParameters final : public torch::nn::Module {
    public:
        PidLagrangianAlphaParameters(
            float k_p, float k_i, float k_d, float initial_alpha, int nb_alphas);

        void update(const torch::Tensor &entropy, const torch::Tensor &target_entropy) const;

        torch::Tensor alpha() const;

    private:
        static constexpr float MIN_ALPHA = 1e-8f;
        static constexpr float MAX_ALPHA = 1.f;

        float k_p, k_i, k_d;

        torch::Tensor previous_entropy;
        torch::Tensor has_previous;

        torch::Tensor integral;
        torch::Tensor log_alpha_tensor;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_ENTROPY_H
