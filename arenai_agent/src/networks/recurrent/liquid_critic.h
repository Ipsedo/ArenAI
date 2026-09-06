//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_LIQUID_CRITIC_H
#define ARENAI_LIQUID_CRITIC_H

#include <memory>

#include <torch/torch.h>

#include "../vision.h"
#include "./liquid_recurrent.h"

namespace arenai::agent {

    struct LiquidCriticOutput {
        torch::Tensor value;
        // liquid state after the last processed step
        torch::Tensor next_x;
    };

    // same encoders as the liquid actor, ending on a state-value head
    class LiquidCritic final : public torch::nn::Module {
    public:
        explicit LiquidCritic(
            const int &vision_height, const int &vision_width, const int &nb_sensors,
            const int &hidden_size_sensors,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, const int &neuron_number,
            const int &unfolding_steps, const float &delta_t);

        // one env step: vision [B, C, H, W], sensors [B, S], x_t [B, neuron_number]
        LiquidCriticOutput
        value(const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &x_t);

        // BPTT chunk: vision [B, T, C, H, W], sensors [B, T, S], x_t [B, neuron_number];
        // value is [B, T, 1]
        LiquidCriticOutput value_sequence(
            const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &x_t);

        torch::Tensor initial_state(int batch_size);

    private:
        std::shared_ptr<ConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;

        std::shared_ptr<LiquidRecurrent> liquid;

        torch::nn::Linear to_value;

        // [rows, features] recurrent input: encoded vision and sensors, concatenated
        torch::Tensor encode(const torch::Tensor &vision, const torch::Tensor &sensors);
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_CRITIC_H
