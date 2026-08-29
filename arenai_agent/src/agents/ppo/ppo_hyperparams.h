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
        int hidden_size_sensors = 128;
        std::vector<int> actor_hidden_sizes = {1024, 512};
        std::vector<int> critic_hidden_sizes = {1024, 512};
        std::vector<std::tuple<int, int>> vision_channels = {{3, 8},   {8, 16},  {16, 24},
                                                             {24, 32}, {32, 48}, {48, 64}};
        std::vector<int> group_norm_nums = {1, 2, 3, 4, 6, 8};
        float initial_sigma = 0.242f;
        float initial_fire_proba = 0.5f;
        int metric_window_size = 256;
        float gamma = 0.997f;
        float gae_lambda = 0.99f;
        float clip_epsilon = 0.2f;
        float target_kl = 0.05f;
        float grad_norm_max = 0.5f;
        float continuous_target_entropy_init = 0.f;
        float continuous_target_entropy_final = -1.f;
        int target_entropy_warmup_steps = 7500000;
        float discrete_entropy_factor_init = 0.98f;
        float discrete_entropy_factor_final = 0.3f;
        int epochs = 2;
        int rollout_size = 30 * 30;
        int minibatch_size = 1024;
    };

    std::vector<CliField<PpoHyperParams>> ppo_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_PPO_HYPERPARAMS_H
