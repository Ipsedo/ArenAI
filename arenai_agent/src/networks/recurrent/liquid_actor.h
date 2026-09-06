//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_LIQUID_ACTOR_H
#define ARENAI_LIQUID_ACTOR_H

#include <memory>

#include <torch/torch.h>

#include "../vision.h"
#include "./liquid_recurrent.h"

namespace arenai::agent {

    struct LiquidActorOutput {
        torch::Tensor mu;
        torch::Tensor sigma;
        torch::Tensor discrete;
        // liquid state after the last processed step
        torch::Tensor next_x;
    };

    // CNN on the frames, MLP on the proprioception, both concatenated into the
    // recurrent input of each step, then run through the liquid network
    class LiquidActor final : public torch::nn::Module {
    public:
        explicit LiquidActor(
            const int &vision_height, const int &vision_width, const int &nb_sensors,
            const int &nb_continuous_actions, const int &nb_discrete_actions,
            const int &hidden_size_sensors,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, const int &neuron_number,
            const int &unfolding_steps, const float &delta_t, const float &initial_sigma,
            const float &initial_fire_proba);

        // one env step: vision [B, C, H, W], sensors [B, S], x_t [B, neuron_number]
        LiquidActorOutput
        act(const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &x_t);

        // BPTT chunk: vision [B, T, C, H, W], sensors [B, T, S], x_t [B, neuron_number];
        // outputs are [B, T, ...]
        LiquidActorOutput act_sequence(
            const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &x_t);

        torch::Tensor initial_state(int batch_size) const;

    private:
        std::shared_ptr<ConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;

        std::shared_ptr<LiquidRecurrent> liquid;

        torch::nn::Sequential mu;
        torch::nn::Sequential sigma;
        torch::nn::Sequential discrete;

        // [rows, features] recurrent input: encoded vision and sensors, concatenated
        torch::Tensor encode(const torch::Tensor &vision, const torch::Tensor &sensors);
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_ACTOR_H
