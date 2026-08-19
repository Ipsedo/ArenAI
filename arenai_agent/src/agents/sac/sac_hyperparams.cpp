//
// Created by claude on 22/07/2026.
//

#include "./sac_hyperparams.h"

namespace arenai::agent {

    std::vector<CliField<SacHyperParams>> sac_cli_fields() {
        return {
            {.name = "--actor_learning_rate", .member = &SacHyperParams::actor_learning_rate},
            {.name = "--critic_learning_rate", .member = &SacHyperParams::critic_learning_rate},
            {.name = "--alpha_learning_rate", .member = &SacHyperParams::alpha_learning_rate},
            {.name = "--hidden_size_sensors", .member = &SacHyperParams::hidden_size_sensors},
            {.name = "--hidden_size_actions", .member = &SacHyperParams::hidden_size_actions},
            {.name = "--actor_hidden_sizes", .member = &SacHyperParams::actor_hidden_sizes},
            {.name = "--critic_hidden_sizes", .member = &SacHyperParams::critic_hidden_sizes},
            {.name = "--vision_channels", .member = &SacHyperParams::vision_channels},
            {.name = "--group_norm_nums", .member = &SacHyperParams::group_norm_nums},
            {.name = "--initial_sigma", .member = &SacHyperParams::initial_sigma},
            {.name = "--initial_fire_proba", .member = &SacHyperParams::initial_fire_proba},
            {.name = "--target_sigma", .member = &SacHyperParams::target_sigma},
            {.name = "--target_fire_proba", .member = &SacHyperParams::target_fire_proba},
            {.name = "--metric_window_size", .member = &SacHyperParams::metric_window_size},
            {.name = "--tau", .member = &SacHyperParams::tau},
            {.name = "--gamma", .member = &SacHyperParams::gamma},
            {.name = "--replay_buffer_size", .member = &SacHyperParams::replay_buffer_size},
            {.name = "--train_every", .member = &SacHyperParams::train_every},
            {.name = "--epochs", .member = &SacHyperParams::epochs},
            {.name = "--batch_size", .member = &SacHyperParams::batch_size},
        };
    }

}// namespace arenai::agent
