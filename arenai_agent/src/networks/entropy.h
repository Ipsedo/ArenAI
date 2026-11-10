#ifndef ARENAI_AGENT_HOST_ENTROPY_H
#define ARENAI_AGENT_HOST_ENTROPY_H

#include <torch/torch.h>

namespace arenai::agent {

    // one learned temperature per action, alpha = exp(log_alpha) kept in [MIN_ALPHA, MAX_ALPHA]
    class AlphaParameters : public torch::nn::Module {
    public:
        static constexpr float MIN_ALPHA = 1e-3f;
        static constexpr float MAX_ALPHA = 1.f;

        AlphaParameters(float initial_alpha, int nb_alphas);

        torch::Tensor alpha() const;

        torch::Tensor loss(const torch::Tensor &entropy_error) const;

        void clamp() const;

    private:
        torch::Tensor log_alpha;
    };

    class SacAlpha final : public torch::nn::Module {
    public:
        SacAlpha(float initial_alpha, int nb_alphas);

        torch::Tensor alpha() const;

    private:
        torch::Tensor log_alpha;
    };

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
