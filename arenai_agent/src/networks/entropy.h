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
        // the current target: pure, safe to call several times inside the same rollout
        virtual torch::Tensor target_entropy() const = 0;

        // advances the schedule by nb_env_steps environment steps. Called once per rollout,
        // so a schedule is expressed in the same unit as the training progress reported in
        // metrics.csv — independent of minibatch_size, epochs, nb_tanks and tank mortality.
        virtual void step(int64_t nb_env_steps);
    };

    /*
     * Constants
     */

    class ConstantTargetEntropy : public AbstractTargetEntropy {
    public:
        explicit ConstantTargetEntropy(float initial_target);

        torch::Tensor target_entropy() const override;

    private:
        torch::Tensor initial_target;
    };

    /*
     * Lagrangian
     */

    // Equality constraint H = target: alpha is a signed multiplier — a bonus while entropy
    // sits under the target, a penalty once it overshoots — so the policy cannot silently
    // re-inflate after a descent (train_375's collapse mode)
    class PidLagrangianAlphaParameters final : public torch::nn::Module {
    public:
        PidLagrangianAlphaParameters(
            float k_p, float k_i, float k_d, float initial_alpha, int nb_alphas);

        void update(const torch::Tensor &entropy, const torch::Tensor &target_entropy) const;

        torch::Tensor alpha() const;

    private:
        static constexpr float MAX_ALPHA_ABS = 1.f;

        float k_p, k_i, k_d;

        torch::Tensor previous_entropy;
        torch::Tensor has_previous;

        torch::Tensor integral;
        torch::Tensor alpha_tensor;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_ENTROPY_H
