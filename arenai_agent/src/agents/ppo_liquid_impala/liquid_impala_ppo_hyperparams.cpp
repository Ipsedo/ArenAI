//
// Created by claude on 27/10/2026.
//

#include "./liquid_impala_ppo_hyperparams.h"

namespace arenai::agent {

    std::vector<CliField<LiquidImpalaPpoHyperParams>> liquid_impala_ppo_cli_fields() {
        return {
            {.name = "--actor_learning_rate",
             .member = &LiquidImpalaPpoHyperParams::actor_learning_rate},
            {.name = "--critic_learning_rate",
             .member = &LiquidImpalaPpoHyperParams::critic_learning_rate},
            {.name = "--hidden_size_sensors",
             .member = &LiquidImpalaPpoHyperParams::hidden_size_sensors},
            {.name = "--hidden_size", .member = &LiquidImpalaPpoHyperParams::hidden_size},
            {.name = "--vision_channels", .member = &LiquidImpalaPpoHyperParams::vision_channels},
            {.name = "--neuron_number", .member = &LiquidImpalaPpoHyperParams::neuron_number},
            {.name = "--unfolding_steps", .member = &LiquidImpalaPpoHyperParams::unfolding_steps},
            {.name = "--delta_t", .member = &LiquidImpalaPpoHyperParams::delta_t},
            {.name = "--chunk_size", .member = &LiquidImpalaPpoHyperParams::chunk_size},
            {.name = "--initial_sigma", .member = &LiquidImpalaPpoHyperParams::initial_sigma},
            {.name = "--initial_fire_proba",
             .member = &LiquidImpalaPpoHyperParams::initial_fire_proba},
            {.name = "--initial_zoom_proba",
             .member = &LiquidImpalaPpoHyperParams::initial_zoom_proba},
            {.name = "--metric_window_size",
             .member = &LiquidImpalaPpoHyperParams::metric_window_size},
            {.name = "--gamma", .member = &LiquidImpalaPpoHyperParams::gamma},
            {.name = "--gae_lambda", .member = &LiquidImpalaPpoHyperParams::gae_lambda},
            {.name = "--clip_epsilon", .member = &LiquidImpalaPpoHyperParams::clip_epsilon},
            {.name = "--target_kl", .member = &LiquidImpalaPpoHyperParams::target_kl},
            {.name = "--grad_norm_max", .member = &LiquidImpalaPpoHyperParams::grad_norm_max},
            {.name = "--continuous_entropy_factor",
             .member = &LiquidImpalaPpoHyperParams::continuous_entropy_factor},
            {.name = "--discrete_entropy_factor",
             .member = &LiquidImpalaPpoHyperParams::discrete_entropy_factor},
            {.name = "--epochs", .member = &LiquidImpalaPpoHyperParams::epochs},
            {.name = "--rollout_size", .member = &LiquidImpalaPpoHyperParams::rollout_size},
            {.name = "--minibatch_size", .member = &LiquidImpalaPpoHyperParams::minibatch_size},
        };
    }

}// namespace arenai::agent
