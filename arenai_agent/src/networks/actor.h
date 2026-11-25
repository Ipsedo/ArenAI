#ifndef ARENAI_AGENT_HOST_ACTOR_H
#define ARENAI_AGENT_HOST_ACTOR_H

#include <memory>
#include <vector>

#include <torch/torch.h>

#include "./misc.h"
#include "./state_encoder.h"
#include "./vision.h"

namespace arenai::agent {

    struct ActorRawOutput {
        torch::Tensor mu;
        torch::Tensor sigma;
        torch::Tensor discrete;
    };

    class Actor : public torch::nn::Module {
    public:
        explicit Actor(
            const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_actions,
            const int &hidden_size_sensors, const std::vector<int> &hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, const float &initial_sigma,
            const std::vector<float> &initial_discrete_probas);
        ActorRawOutput act(const torch::Tensor &vision, const torch::Tensor &sensors);

    private:
        std::shared_ptr<ConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;

        torch::nn::Sequential head;

        torch::nn::Linear mu;
        torch::nn::Sequential sigma;
        torch::nn::Sequential discrete;

    protected:
        torch::Tensor trunc_forward(const torch::Tensor &vision, const torch::Tensor &sensors);
    };

    class DistillationActor final : public Actor {
    public:
        explicit DistillationActor(
            const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_actions,
            const int &hidden_size_sensors, const std::vector<int> &hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, const float &initial_sigma,
            const std::vector<float> &initial_discrete_probas);

        torch::Tensor value(const torch::Tensor &vision, const torch::Tensor &sensors);

    private:
        torch::nn::Linear to_value;
    };

    // reads the state encoding of the critic, only its head is trained by the actor loss
    class SacActor final : public torch::nn::Module {
    public:
        SacActor(
            const std::shared_ptr<StateEncoder> &encoder, int nb_continuous_actions,
            int nb_discrete_actions, const std::vector<int> &hidden_sizes, float initial_sigma,
            const std::vector<float> &initial_discrete_probas);

        ActorRawOutput act(const torch::Tensor &vision, const torch::Tensor &sensors);
        ActorRawOutput act_from_encoding(const torch::Tensor &encoding);

        std::vector<torch::Tensor> head_parameters() const;

    private:
        std::shared_ptr<StateEncoder> encoder;

        torch::nn::Sequential head;

        torch::nn::Linear mu;
        torch::nn::Sequential sigma;
        torch::nn::Sequential discrete;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_ACTOR_H
