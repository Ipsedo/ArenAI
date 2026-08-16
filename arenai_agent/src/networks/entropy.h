//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_AGENT_HOST_ENTROPY_H
#define ARENAI_AGENT_HOST_ENTROPY_H

#include <torch/torch.h>

namespace arenai::agent {

    // one independent coefficient per action dimension: a single coefficient can only
    // constrain the summed entropy, which a wide dimension keeps on target while another one
    // collapses.
    // The bounds are anti-windup guards: dual ascent is a pure integrator, so while the
    // entropy stays on one side of its target log_alpha drifts without bound and ends up
    // orders of magnitude away from any useful value, unable to come back in time once the
    // error flips sign.
    class AlphaParameters : public torch::nn::Module {
    public:
        explicit AlphaParameters(float initial_alpha, int nb_alphas);

        // clamps the parameters in place instead of returning a clamped view: torch::clamp
        // has a zero gradient outside the bounds, so a clamped view would freeze log_alpha
        // for good the first time it steps past a bound, in both directions
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

    class ConstantDiscreteTargetEntropy : public ConstantTargetEntropy {
    public:
        explicit ConstantDiscreteTargetEntropy(float fire_probability);
    };

    class ConstantContinuousTargetEntropy : public ConstantTargetEntropy {
    public:
        explicit ConstantContinuousTargetEntropy(int nb_continuous_action, float target_sigma);
    };

    // target of a single action instead of the sum over them, to be broadcast against one
    // alpha per dimension
    class ConstantContinuousPerActionTargetEntropy : public ConstantTargetEntropy {
    public:
        explicit ConstantContinuousPerActionTargetEntropy(float target_sigma);
    };

    /*
     * Warmup
     */

    class AbstractCosineAnnealingTargetEntropy : public AbstractTargetEntropy {
    public:
        AbstractCosineAnnealingTargetEntropy(
            float initial_value, float final_value, int warmup_step);
        torch::Tensor target_entropy() override;

    protected:
        virtual float to_target_entropy(float value) = 0;

    private:
        float initial;
        float final;
        int warmup_step;

        torch::Tensor current_step;
    };

    class DiscreteCosineAnnealingTargetEntropy : public AbstractCosineAnnealingTargetEntropy {
    public:
        DiscreteCosineAnnealingTargetEntropy(
            float initial_probability, float final_probability, int warmup_step);

    protected:
        float to_target_entropy(float value) override;
    };

    class ContinuousCosineAnnealingTargetEntropy : public AbstractCosineAnnealingTargetEntropy {
    public:
        ContinuousCosineAnnealingTargetEntropy(
            int nb_actions, float initial_sigma, float final_sigma, int warmup_step);

    protected:
        float to_target_entropy(float value) override;

    private:
        int nb_actions;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_ENTROPY_H
