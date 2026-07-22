//
// Created by claude on 22/07/2026.
//

#include "./ppo_hyperparams.h"

namespace arenai::agent {

    std::vector<CliField<PpoHyperParams>> ppo_cli_fields() {
        return {
            {"--actor_learning_rate", &PpoHyperParams::actor_learning_rate},
            {"--critic_learning_rate", &PpoHyperParams::critic_learning_rate},
            {"--hidden_size_sensors", &PpoHyperParams::hidden_size_sensors},
            {"--actor_hidden_sizes", &PpoHyperParams::actor_hidden_sizes},
            {"--critic_hidden_sizes", &PpoHyperParams::critic_hidden_sizes},
            {"--vision_channels", &PpoHyperParams::vision_channels},
            {"--group_norm_nums", &PpoHyperParams::group_norm_nums},
            {"--metric_window_size", &PpoHyperParams::metric_window_size},
            {"--gamma", &PpoHyperParams::gamma},
            {"--gae_lambda", &PpoHyperParams::gae_lambda},
            {"--clip_epsilon", &PpoHyperParams::clip_epsilon},
            {"--continuous_entropy_coef", &PpoHyperParams::continuous_entropy_coef},
            {"--discrete_entropy_coef", &PpoHyperParams::discrete_entropy_coef},
            {"--epochs", &PpoHyperParams::epochs},
            {"--rollout_size", &PpoHyperParams::rollout_size},
        };
    }

}// namespace arenai::agent
