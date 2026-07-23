//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_PPO_HYPERPARAMS_H
#define ARENAI_PPO_HYPERPARAMS_H

#include <tuple>
#include <vector>

#include "../../utils/cli_fields.h"

namespace arenai::agent {

    // Member initializers are the CLI defaults (single source of truth).
    struct PpoHyperParams {
        float actor_learning_rate = 1e-4f;
        float critic_learning_rate = 3e-4f;
        int hidden_size_sensors = 256;
        std::vector<int> actor_hidden_sizes = {2560, 1280};
        std::vector<int> critic_hidden_sizes = {2560, 1280};
        std::vector<std::tuple<int, int>> vision_channels = {{3, 8},   {8, 16},   {16, 32},
                                                             {32, 64}, {64, 128}, {128, 256}};
        std::vector<int> group_norm_nums = {1, 2, 4, 8, 16, 32};
        int metric_window_size = 256;
        float gamma = 0.995f;
        float gae_lambda = 0.95f;
        float clip_epsilon = 0.2f;
        float grad_norm_max = 1.f;
        float continuous_entropy_coef = 0.01f;
        float discrete_entropy_coef = 0.01f;
        int epochs = 4;
        int rollout_size = 300;// 10 s of game time at 30 Hz
        int minibatch_size = 256;
    };

    std::vector<CliField<PpoHyperParams>> ppo_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_PPO_HYPERPARAMS_H
