//
// Created by claude on 26/10/2026.
//

#ifndef ARENAI_AGENT_HOST_IMPALA_ACTOR_H
#define ARENAI_AGENT_HOST_IMPALA_ACTOR_H

#include <memory>
#include <vector>

#include <torch/torch.h>

#include "./actor.h"
#include "./vision.h"

namespace arenai::agent {

    // Actor whose vision is encoded by an IMPALA residual CNN instead of the
    // plain strided convolutions; same heads and same raw output
    class ImpalaActor final : public torch::nn::Module {
    public:
        explicit ImpalaActor(
            const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_actions,
            const int &hidden_size_sensors, const std::vector<int> &hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels, const float &initial_sigma,
            const std::vector<float> &initial_discrete_probas);
        ActorRawOutput act(const torch::Tensor &vision, const torch::Tensor &sensors);

    private:
        std::shared_ptr<ImpalaConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;

        torch::nn::Sequential head;

        torch::nn::Sequential mu;
        torch::nn::Sequential sigma;
        torch::nn::Sequential discrete;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_IMPALA_ACTOR_H
