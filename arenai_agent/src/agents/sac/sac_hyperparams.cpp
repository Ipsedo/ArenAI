//
// Created by claude on 08/11/2026.
//

#include "./sac_hyperparams.h"

namespace arenai::agent {

    std::vector<CliField<SacHyperParams>> sac_cli_fields() {
        return {
            {.name = "--actor_learning_rate", .member = &SacHyperParams::actor_learning_rate},
            {.name = "--critic_learning_rate", .member = &SacHyperParams::critic_learning_rate},
            {.name = "--alpha_learning_rate", .member = &SacHyperParams::alpha_learning_rate},
            {.name = "--hidden_size_sensors", .member = &SacHyperParams::hidden_size_sensors},
            {.name = "--actor_hidden_sizes", .member = &SacHyperParams::actor_hidden_sizes},
            {.name = "--critic_hidden_sizes", .member = &SacHyperParams::critic_hidden_sizes},
            {.name = "--vision_channels", .member = &SacHyperParams::vision_channels},
            {.name = "--group_norm_nums", .member = &SacHyperParams::group_norm_nums},
            {.name = "--initial_sigma", .member = &SacHyperParams::initial_sigma},
            {.name = "--initial_fire_proba", .member = &SacHyperParams::initial_fire_proba},
            {.name = "--initial_zoom_proba", .member = &SacHyperParams::initial_zoom_proba},
            {.name = "--metric_window_size", .member = &SacHyperParams::metric_window_size},
            {.name = "--gamma", .member = &SacHyperParams::gamma},
            {.name = "--tau", .member = &SacHyperParams::tau},
            {.name = "--initial_alpha", .member = &SacHyperParams::initial_alpha},
            {.name = "--target_sigma", .member = &SacHyperParams::target_sigma},
            {.name = "--discrete_target_entropy_ratio",
             .member = &SacHyperParams::discrete_target_entropy_ratio},
            {.name = "--buffer_size", .member = &SacHyperParams::buffer_size},
            {.name = "--armed_fraction", .member = &SacHyperParams::armed_fraction},
            {.name = "--learning_starts", .member = &SacHyperParams::learning_starts},
            {.name = "--gradient_steps", .member = &SacHyperParams::gradient_steps},
            {.name = "--batch_size", .member = &SacHyperParams::batch_size},
            {.name = "--gpu_batch_size", .member = &SacHyperParams::gpu_batch_size},
        };
    }

}// namespace arenai::agent
