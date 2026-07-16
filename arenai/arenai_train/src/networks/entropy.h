//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_ENTROPY_H
#define ARENAI_TRAIN_HOST_ENTROPY_H

#include <torch/torch.h>

namespace arenai::train {

    class AlphaParameter final : public torch::nn::Module {
    public:
        explicit AlphaParameter(float initial_alpha);

        torch::Tensor log_alpha();
        torch::Tensor alpha();

    private:
        torch::Tensor log_alpha_tensor;
    };

    class AbstractTargetEntropy : public torch::nn::Module {
    public:
        virtual void step() = 0;
        virtual torch::Tensor target_entropy() = 0;
    };

    class ConstantTargetEntropy : public AbstractTargetEntropy {
    public:
        explicit ConstantTargetEntropy(float initial_target);

        void step() override;

        torch::Tensor target_entropy() override;

    private:
        torch::Tensor initial_target;
    };

    class AbstractTargetEntropyWarmup : public AbstractTargetEntropy {
    public:
        AbstractTargetEntropyWarmup(float initial_value, float final_value, int warmup_step);

        void step() override;
        torch::Tensor target_entropy() override;

    protected:
        virtual float to_target_entropy(float value) = 0;

    private:
        float initial;
        float final;
        int warmup_step;

        torch::Tensor current_step;
    };

    class DiscreteTargetEntropyWarmup : public AbstractTargetEntropyWarmup {
    public:
        DiscreteTargetEntropyWarmup(
            int nb_actions, float initial_factor, float final_factor, int warmup_step);

    protected:
        float to_target_entropy(float value) override;

    private:
        int nb_actions;
    };

    class ContinuousTargetEntropyWarmup : public AbstractTargetEntropyWarmup {
    public:
        ContinuousTargetEntropyWarmup(
            int nb_actions, float initial_sigma, float final_sigma, int warmup_step);

    protected:
        float to_target_entropy(float value) override;

    private:
        int nb_actions;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_ENTROPY_H
