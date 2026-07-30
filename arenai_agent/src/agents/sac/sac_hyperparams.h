//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_SAC_HYPERPARAMS_H
#define ARENAI_SAC_HYPERPARAMS_H

#include <tuple>
#include <vector>

#include "../../utils/cli_fields.h"

namespace arenai::agent {

    // Member initializers are the CLI defaults (single source of truth).
    struct SacHyperParams {
        float actor_learning_rate = 1e-4f;
        float critic_learning_rate = 3e-4f;
        float alpha_learning_rate = 3e-4f;
        int hidden_size_sensors = 128;
        int hidden_size_actions = 32;
        std::vector<int> actor_hidden_sizes = {1024, 512};
        std::vector<int> critic_hidden_sizes = {1024, 512};
        std::vector<std::tuple<int, int>> vision_channels = {{3, 8}, {8, 16}, {16, 32}, {32, 64}};
        std::vector<int> group_norm_nums = {1, 2, 4, 8};
        int metric_window_size = 256;
        float tau = 0.005f;
        float gamma = 0.995f;
        int replay_buffer_size = 5000000;
        int train_every = 256;
        int epochs = 16;
        int batch_size = 8192;
    };

    std::vector<CliField<SacHyperParams>> sac_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_SAC_HYPERPARAMS_H
