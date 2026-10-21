//
// Created by samuel on 21/10/2026.
//

#ifndef ARENAI_RETURN_SCALER_H
#define ARENAI_RETURN_SCALER_H

#include <torch/torch.h>

namespace arenai::agent {

    // Divides the rewards by the running standard deviation of the discounted return.
    // No mean shift: the reward signs and the terminal semantics are kept.
    class ReturnScaler final : public torch::nn::Module {
    public:
        explicit ReturnScaler(float gamma);

        // rewards / dones / valids: [T, nb_tanks, 1] on CPU; the statistics are
        // updated with this rollout's discounted returns before scaling
        torch::Tensor scale(
            const torch::Tensor &rewards, const torch::Tensor &dones, const torch::Tensor &valids);

    private:
        float gamma;

        // running statistics of the discounted return (saved with the checkpoint)
        torch::Tensor count;
        torch::Tensor mean;
        torch::Tensor var;

        // [nb_tanks, 1] discounted return accumulated so far, carried across rollouts
        // and cut at each done
        torch::Tensor discounted_returns;
    };

}// namespace arenai::agent

#endif//ARENAI_RETURN_SCALER_H
