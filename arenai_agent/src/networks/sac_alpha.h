//
// Created by claude on 10/11/2026.
//

#ifndef ARENAI_SAC_ALPHA_H
#define ARENAI_SAC_ALPHA_H

#include <torch/torch.h>

namespace arenai::agent {

    // SAC temperatures, one per action: alpha = exp(log_alpha), learned by dual ascent on
    // log_alpha and deliberately unbounded — unlike the on-policy AlphaParameters, the
    // temperature has to decay as far as the critic needs (run 414 only learns under ~5e-4)
    class SacAlpha final : public torch::nn::Module {
    public:
        SacAlpha(float initial_alpha, int nb_alphas);

        torch::Tensor alpha() const;

    private:
        torch::Tensor log_alpha;
    };

}// namespace arenai::agent

#endif//ARENAI_SAC_ALPHA_H
