//
// Created by samuel on 23/10/2025.
//

#ifndef ARENAI_AGENT_HOST_ACTOR_H
#define ARENAI_AGENT_HOST_ACTOR_H

#include <memory>

#include <torch/torch.h>

#include "./vision.h"

namespace arenai::agent {

    struct ActorRawOutput {
        // Beta distribution on [-1, 1]: mode in [0, 1] (on the underlying [0, 1]
        // support) and concentration κ = α + β
        torch::Tensor mode;
        torch::Tensor concentration;
        torch::Tensor discrete;
    };

    class Actor final : public torch::nn::Module {
    public:
        explicit Actor(
            const int &vision_height, const int &vision_width, const int &nb_sensors,
            const int &nb_continuous_actions, const int &nb_discrete_actions,
            const int &hidden_size_sensors, const std::vector<int> &hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, const float &initial_sigma,
            const float &initial_fire_proba);
        ActorRawOutput act(const torch::Tensor &vision, const torch::Tensor &sensors);

    private:
        std::shared_ptr<ConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;

        torch::nn::Sequential head;

        torch::nn::Sequential mode;
        torch::nn::Sequential concentration;
        torch::nn::Sequential discrete;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_ACTOR_H
