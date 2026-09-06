//
// Created by samuel on 06/09/2026.
//

#include "./liquid_ppo_hyperparams.h"

namespace arenai::agent {

    std::vector<CliField<LiquidPpoHyperParams>> liquid_ppo_cli_fields() {
        return {
            {.name = "--actor_learning_rate", .member = &LiquidPpoHyperParams::actor_learning_rate},
            {.name = "--critic_learning_rate",
             .member = &LiquidPpoHyperParams::critic_learning_rate},
            {.name = "--hidden_size_sensors", .member = &LiquidPpoHyperParams::hidden_size_sensors},
            {.name = "--vision_channels", .member = &LiquidPpoHyperParams::vision_channels},
            {.name = "--group_norm_nums", .member = &LiquidPpoHyperParams::group_norm_nums},
            {.name = "--neuron_number", .member = &LiquidPpoHyperParams::neuron_number},
            {.name = "--unfolding_steps", .member = &LiquidPpoHyperParams::unfolding_steps},
            {.name = "--delta_t", .member = &LiquidPpoHyperParams::delta_t},
            {.name = "--chunk_size", .member = &LiquidPpoHyperParams::chunk_size},
            {.name = "--initial_sigma", .member = &LiquidPpoHyperParams::initial_sigma},
            {.name = "--initial_fire_proba", .member = &LiquidPpoHyperParams::initial_fire_proba},
            {.name = "--metric_window_size", .member = &LiquidPpoHyperParams::metric_window_size},
            {.name = "--gamma", .member = &LiquidPpoHyperParams::gamma},
            {.name = "--gae_lambda", .member = &LiquidPpoHyperParams::gae_lambda},
            {.name = "--clip_epsilon", .member = &LiquidPpoHyperParams::clip_epsilon},
            {.name = "--target_kl", .member = &LiquidPpoHyperParams::target_kl},
            {.name = "--grad_norm_max", .member = &LiquidPpoHyperParams::grad_norm_max},
            {.name = "--continuous_target_entropy",
             .member = &LiquidPpoHyperParams::continuous_target_entropy},
            {.name = "--discrete_target_entropy_factor",
             .member = &LiquidPpoHyperParams::discrete_target_entropy_factor},
            {.name = "--epochs", .member = &LiquidPpoHyperParams::epochs},
            {.name = "--rollout_size", .member = &LiquidPpoHyperParams::rollout_size},
            {.name = "--minibatch_size", .member = &LiquidPpoHyperParams::minibatch_size},
        };
    }

}// namespace arenai::agent
