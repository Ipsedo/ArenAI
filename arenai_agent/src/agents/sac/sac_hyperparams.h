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
        float alpha_learning_rate = 3e-5f;
        int hidden_size_sensors = 128;
        int hidden_size_actions = 32;
        std::vector<int> actor_hidden_sizes = {1024, 512};
        std::vector<int> critic_hidden_sizes = {1024, 512};
        std::vector<std::tuple<int, int>> vision_channels = {{3, 8},   {8, 16},  {16, 24},
                                                             {24, 32}, {32, 48}, {48, 64}};
        std::vector<int> group_norm_nums = {1, 2, 3, 4, 6, 8};
        float initial_sigma = 0.5f;
        float initial_fire_proba = 0.5f;
        float continuous_target_entropy = -1.f;
        float discrete_target_entropy_factor = 0.3f;
        int metric_window_size = 256;
        float tau = 0.005f;
        float gamma = 0.997f;
        int replay_buffer_size = 300000;
        int train_every = 256;
        int epochs = 128;
        int batch_size = 256;
    };

    std::vector<CliField<SacHyperParams>> sac_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_SAC_HYPERPARAMS_H
