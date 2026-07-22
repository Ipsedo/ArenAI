//
// Created by claude on 22/07/2026.
//

#include "./sac_hyperparams.h"

namespace arenai::agent {

    std::vector<CliField<SacHyperParams>> sac_cli_fields() {
        return {
            {"--actor_learning_rate", &SacHyperParams::actor_learning_rate},
            {"--critic_learning_rate", &SacHyperParams::critic_learning_rate},
            {"--alpha_learning_rate", &SacHyperParams::alpha_learning_rate},
            {"--hidden_size_sensors", &SacHyperParams::hidden_size_sensors},
            {"--hidden_size_actions", &SacHyperParams::hidden_size_actions},
            {"--actor_hidden_sizes", &SacHyperParams::actor_hidden_sizes},
            {"--critic_hidden_sizes", &SacHyperParams::critic_hidden_sizes},
            {"--vision_channels", &SacHyperParams::vision_channels},
            {"--group_norm_nums", &SacHyperParams::group_norm_nums},
            {"--metric_window_size", &SacHyperParams::metric_window_size},
            {"--tau", &SacHyperParams::tau},
            {"--gamma", &SacHyperParams::gamma},
            {"--replay_buffer_size", &SacHyperParams::replay_buffer_size},
            {"--train_every", &SacHyperParams::train_every},
            {"--epochs", &SacHyperParams::epochs},
            {"--batch_size", &SacHyperParams::batch_size},
        };
    }

}// namespace arenai::agent
