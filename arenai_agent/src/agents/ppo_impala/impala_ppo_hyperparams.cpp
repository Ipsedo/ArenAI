//
// Created by claude on 26/10/2026.
//

#include "./impala_ppo_hyperparams.h"

namespace arenai::agent {

    std::vector<CliField<ImpalaPpoHyperParams>> impala_ppo_cli_fields() {
        return {
            {.name = "--actor_learning_rate", .member = &ImpalaPpoHyperParams::actor_learning_rate},
            {.name = "--critic_learning_rate",
             .member = &ImpalaPpoHyperParams::critic_learning_rate},
            {.name = "--hidden_size_sensors", .member = &ImpalaPpoHyperParams::hidden_size_sensors},
            {.name = "--actor_hidden_sizes", .member = &ImpalaPpoHyperParams::actor_hidden_sizes},
            {.name = "--critic_hidden_sizes", .member = &ImpalaPpoHyperParams::critic_hidden_sizes},
            {.name = "--vision_channels", .member = &ImpalaPpoHyperParams::vision_channels},
            {.name = "--initial_sigma", .member = &ImpalaPpoHyperParams::initial_sigma},
            {.name = "--initial_fire_proba", .member = &ImpalaPpoHyperParams::initial_fire_proba},
            {.name = "--initial_zoom_proba", .member = &ImpalaPpoHyperParams::initial_zoom_proba},
            {.name = "--metric_window_size", .member = &ImpalaPpoHyperParams::metric_window_size},
            {.name = "--gamma", .member = &ImpalaPpoHyperParams::gamma},
            {.name = "--gae_lambda", .member = &ImpalaPpoHyperParams::gae_lambda},
            {.name = "--clip_epsilon", .member = &ImpalaPpoHyperParams::clip_epsilon},
            {.name = "--target_kl", .member = &ImpalaPpoHyperParams::target_kl},
            {.name = "--grad_norm_max", .member = &ImpalaPpoHyperParams::grad_norm_max},
            {.name = "--continuous_entropy_factor",
             .member = &ImpalaPpoHyperParams::continuous_entropy_factor},
            {.name = "--discrete_entropy_factor",
             .member = &ImpalaPpoHyperParams::discrete_entropy_factor},
            {.name = "--epochs", .member = &ImpalaPpoHyperParams::epochs},
            {.name = "--rollout_size", .member = &ImpalaPpoHyperParams::rollout_size},
            {.name = "--minibatch_size", .member = &ImpalaPpoHyperParams::minibatch_size},
        };
    }

}// namespace arenai::agent
