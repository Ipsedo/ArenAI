//
// Created by claude on 22/07/2026.
//

#include "./ppo_hyperparams.h"

namespace arenai::agent {

    std::vector<CliField<PpoHyperParams>> ppo_cli_fields() {
        return {
            {.name = "--actor_learning_rate", .member = &PpoHyperParams::actor_learning_rate},
            {.name = "--critic_learning_rate", .member = &PpoHyperParams::critic_learning_rate},
            {.name = "--hidden_size_sensors", .member = &PpoHyperParams::hidden_size_sensors},
            {.name = "--actor_hidden_sizes", .member = &PpoHyperParams::actor_hidden_sizes},
            {.name = "--critic_hidden_sizes", .member = &PpoHyperParams::critic_hidden_sizes},
            {.name = "--vision_channels", .member = &PpoHyperParams::vision_channels},
            {.name = "--group_norm_nums", .member = &PpoHyperParams::group_norm_nums},
            {.name = "--initial_sigma", .member = &PpoHyperParams::initial_sigma},
            {.name = "--initial_fire_proba", .member = &PpoHyperParams::initial_fire_proba},
            {.name = "--metric_window_size", .member = &PpoHyperParams::metric_window_size},
            {.name = "--gamma", .member = &PpoHyperParams::gamma},
            {.name = "--gae_lambda", .member = &PpoHyperParams::gae_lambda},
            {.name = "--clip_epsilon", .member = &PpoHyperParams::clip_epsilon},
            {.name = "--target_kl", .member = &PpoHyperParams::target_kl},
            {.name = "--grad_norm_max", .member = &PpoHyperParams::grad_norm_max},
            {.name = "--continuous_entropy_coefficient",
             .member = &PpoHyperParams::continuous_entropy_coefficient},
            {.name = "--discrete_entropy_coefficient",
             .member = &PpoHyperParams::discrete_entropy_coefficient},
            {.name = "--epochs", .member = &PpoHyperParams::epochs},
            {.name = "--rollout_size", .member = &PpoHyperParams::rollout_size},
            {.name = "--minibatch_size", .member = &PpoHyperParams::minibatch_size},
        };
    }

}// namespace arenai::agent
