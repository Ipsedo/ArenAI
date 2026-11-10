#ifndef ARENAI_RETURN_SCALER_H
#define ARENAI_RETURN_SCALER_H

#include <torch/torch.h>

namespace arenai::agent {

    class ReturnScaler final : public torch::nn::Module {
    public:
        explicit ReturnScaler(float gamma);

        torch::Tensor scale(
            const torch::Tensor &rewards, const torch::Tensor &dones, const torch::Tensor &valids);

    private:
        float gamma;

        torch::Tensor count;
        torch::Tensor mean;
        torch::Tensor var;

        torch::Tensor discounted_returns;
    };

}// namespace arenai::agent

#endif//ARENAI_RETURN_SCALER_H
