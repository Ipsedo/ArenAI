//
// Created by claude on 27/10/2026.
//

#ifndef ARENAI_LIQUID_IMPALA_CRITIC_H
#define ARENAI_LIQUID_IMPALA_CRITIC_H

#include <memory>

#include <torch/torch.h>

#include "../vision.h"
#include "./liquid_cell.h"
#include "./liquid_critic.h"

namespace arenai::agent {

    // same encoders as the liquid IMPALA actor, ending on a state-value head
    class LiquidImpalaCritic final : public torch::nn::Module {
    public:
        explicit LiquidImpalaCritic(
            const int &nb_sensors, const int &hidden_size_sensors, const int &hidden_size,
            const std::vector<std::tuple<int, int>> &vision_channels, const int &neuron_number,
            const int &unfolding_steps, const float &delta_t);

        // one env step: vision [B, C, H, W], sensors [B, S], x_t [B, neuron_number]
        LiquidCriticOutput
        value(const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &x_t);

        // BPTT chunk: vision [B, T, C, H, W], sensors [B, T, S], x_t [B, neuron_number];
        // value is [B, T, 1]
        LiquidCriticOutput value_sequence(
            const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &x_t);

        torch::Tensor initial_state(int batch_size) const;

    private:
        std::shared_ptr<ImpalaConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;

        std::shared_ptr<LiquidCell> liquid;

        torch::nn::Sequential to_value;

        // [rows, features] recurrent input: encoded vision and sensors, concatenated
        torch::Tensor encode(const torch::Tensor &vision, const torch::Tensor &sensors);
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_IMPALA_CRITIC_H
