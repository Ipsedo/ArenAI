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
        int metric_window_size = 256;
        // 0.997 at 30 Hz -> ~11 s credit horizon (shell flight time + fights stay visible)
        float gamma = 0.997f;
        // 0.98: a shell resolving 30-60 steps after the fire still reaches the fire
        // decision at x0.25-0.5 through GAE, instead of x0.04-0.2 with 0.95
        float gae_lambda = 0.98f;
        float clip_epsilon = 0.2f;
        // early-stop of the epoch loop when approx KL > 1.5 * target_kl; <= 0 disables it
        float target_kl = 0.02f;
        float grad_norm_max = 0.5f;
        float continuous_alpha_learning_rate = 4e-2f;
        float discrete_alpha_learning_rate = 1e-1f;
        int epochs = 4;
        int rollout_size = 30 * 30;
        int minibatch_size = 1024;
    };

    std::vector<CliField<PpoHyperParams>> ppo_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_PPO_HYPERPARAMS_H
