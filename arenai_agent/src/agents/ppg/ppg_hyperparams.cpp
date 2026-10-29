//
// Created by samuel on 22/07/2026.
//

#include "./ppg_hyperparams.h"

namespace arenai::agent {

    std::vector<CliField<PpgHyperParams>> ppg_cli_fields() {
        return {
            {.name = "--actor_learning_rate", .member = &PpgHyperParams::actor_learning_rate},
            {.name = "--critic_learning_rate", .member = &PpgHyperParams::critic_learning_rate},
            {.name = "--hidden_size_sensors", .member = &PpgHyperParams::hidden_size_sensors},
            {.name = "--actor_hidden_sizes", .member = &PpgHyperParams::actor_hidden_sizes},
            {.name = "--critic_hidden_sizes", .member = &PpgHyperParams::critic_hidden_sizes},
            {.name = "--vision_channels", .member = &PpgHyperParams::vision_channels},
            {.name = "--group_norm_nums", .member = &PpgHyperParams::group_norm_nums},
            {.name = "--initial_sigma", .member = &PpgHyperParams::initial_sigma},
            {.name = "--initial_fire_proba", .member = &PpgHyperParams::initial_fire_proba},
            {.name = "--initial_zoom_proba", .member = &PpgHyperParams::initial_zoom_proba},
            {.name = "--metric_window_size", .member = &PpgHyperParams::metric_window_size},
            {.name = "--gamma", .member = &PpgHyperParams::gamma},
            {.name = "--gae_lambda", .member = &PpgHyperParams::gae_lambda},
            {.name = "--clip_epsilon", .member = &PpgHyperParams::clip_epsilon},
            {.name = "--target_kl", .member = &PpgHyperParams::target_kl},
            {.name = "--grad_norm_max", .member = &PpgHyperParams::grad_norm_max},
            {.name = "--continuous_entropy_factor",
             .member = &PpgHyperParams::continuous_entropy_factor},
            {.name = "--discrete_entropy_factor",
             .member = &PpgHyperParams::discrete_entropy_factor},
            {.name = "--actor_epochs", .member = &PpgHyperParams::actor_epochs},
            {.name = "--critic_epochs", .member = &PpgHyperParams::critic_epochs},
            {.name = "--rollout_size", .member = &PpgHyperParams::rollout_size},
            {.name = "--actor_batch_size", .member = &PpgHyperParams::actor_batch_size},
            {.name = "--actor_gpu_batch_size", .member = &PpgHyperParams::actor_gpu_batch_size},
            {.name = "--critic_batch_size", .member = &PpgHyperParams::critic_batch_size},
            {.name = "--critic_gpu_batch_size", .member = &PpgHyperParams::critic_gpu_batch_size},
            {.name = "--actor_aux_epochs", .member = &PpgHyperParams::actor_aux_epochs},
            {.name = "--critic_aux_epochs", .member = &PpgHyperParams::critic_aux_epochs},
            {.name = "--beta_clone", .member = &PpgHyperParams::beta_clone},
        };
    }

}// namespace arenai::agent
