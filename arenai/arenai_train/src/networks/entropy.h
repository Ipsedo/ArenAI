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

    class TargetEntropyWarmup final : public torch::nn::Module {
    public:
        TargetEntropyWarmup(
            float initial_target_entropy, float final_target_entropy, int warmup_step);

        void step();
        torch::Tensor target_entropy() const;

    private:
        float initial;
        float final;
        int warmup_step;

        torch::Tensor current_step;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_ENTROPY_H
