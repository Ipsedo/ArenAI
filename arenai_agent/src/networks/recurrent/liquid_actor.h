#ifndef ARENAI_LIQUID_ACTOR_H
#define ARENAI_LIQUID_ACTOR_H

#include <memory>

#include <torch/torch.h>

#include "../../distributions/spatial_aim_law.h"
#include "../aim_head.h"
#include "../misc.h"
#include "../presence_model.h"
#include "../vision.h"
#include "./liquid_cell.h"

namespace arenai::agent {

    struct LiquidActorOutput {
        // drive actions
        torch::Tensor mu;
        torch::Tensor sigma;
        // independent Bernoulli probabilities, one per discrete action
        torch::Tensor discrete;
        // liquid state after the last processed step
        torch::Tensor next_x;
        torch::Tensor features;
        torch::Tensor vision_features;
        // over the rows [B] of act, the flattened rows [B * T] of act_sequence
        SpatialAimLaw aim;
    };

    class LiquidActor final : public torch::nn::Module, public PresenceModel {
    public:
        explicit LiquidActor(
            const int &vision_height, const int &vision_width, const int &nb_sensors,
            const int &nb_continuous_actions, const int &nb_discrete_actions,
            const int &hidden_size_sensors,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, const int &neuron_number,
            const int &unfolding_steps, const float &delta_t, const float &initial_sigma,
            const std::vector<float> &initial_discrete_probas);

        // one env step: vision [B, C, H, W], sensors [B, S], x_t [B, neuron_number]
        LiquidActorOutput
        act(const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &x_t);

        LiquidActorOutput act_sequence(
            const torch::Tensor &vision, const torch::Tensor &sensors, const torch::Tensor &x_t);

        torch::Tensor initial_state(int batch_size) const;

        torch::Tensor presence_logits(const torch::Tensor &vision) override;
        std::vector<torch::Tensor> presence_parameters() const override;

    private:
        std::shared_ptr<ConvolutionNetwork> vision_encoder;
        torch::nn::Sequential sensors_encoder;

        std::shared_ptr<LiquidCell> liquid;

        torch::nn::Linear mu;
        torch::nn::Sequential sigma;
        torch::nn::Sequential discrete;
        std::shared_ptr<AimHead> aim;

        // recurrent input [rows, features]: encoded vision and sensors, concatenated
        std::tuple<torch::Tensor, torch::Tensor>
        encode(const torch::Tensor &vision, const torch::Tensor &sensors);
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_ACTOR_H
