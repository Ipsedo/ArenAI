//
// Created by claude on 08/11/2026.
//

#ifndef ARENAI_SAC_ACTOR_H
#define ARENAI_SAC_ACTOR_H

#include <memory>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "./vision.h"

namespace arenai::agent {

    struct SacActorOutput {
        // tanh-squashed gaussian: mu and sigma are the pre-squash mean and spread, both
        // state-dependent (log-sigma is kept in [LOG_SIGMA_MIN, LOG_SIGMA_MAX])
        torch::Tensor mu;
        torch::Tensor sigma;
        // [N, nb_discrete_actions] independent Bernoulli probabilities
        torch::Tensor discrete;
    };

    class SacActor final : public torch::nn::Module {
    public:
        SacActor(
            int nb_sensors, int nb_continuous_actions, int nb_discrete_actions,
            int hidden_size_sensors, const std::vector<int> &hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, float initial_sigma,
            const std::vector<float> &initial_discrete_probas);

        SacActorOutput act(const torch::Tensor &vision, const torch::Tensor &sensors);

    private:
        std::shared_ptr<ConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;
        torch::nn::Sequential head;

        torch::nn::Linear mu;
        torch::nn::Linear log_sigma;
        torch::nn::Sequential discrete;
    };

}// namespace arenai::agent

#endif//ARENAI_SAC_ACTOR_H
