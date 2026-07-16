//
// Created by samuel on 19/05/2026.
//

#ifndef ARENAI_TRAIN_HOST_MISC_H
#define ARENAI_TRAIN_HOST_MISC_H

#include <torch/torch.h>

namespace arenai::train {

    class Clamp : public torch::nn::Module {
    public:
        Clamp(float lower_bound, float upper_bound);

        torch::Tensor forward(const torch::Tensor &x);

        void pretty_print(std::ostream &stream) const override;

    private:
        float lower_bound;
        float upper_bound;
    };

    class Exp : public torch::nn::Module {
    public:
        torch::Tensor forward(const torch::Tensor &x);

        void pretty_print(std::ostream &stream) const override;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_MISC_H
