//
// Created by claude on 26/10/2026.
//

#ifndef ARENAI_IMPALA_PPO_HYPERPARAMS_H
#define ARENAI_IMPALA_PPO_HYPERPARAMS_H

#include <tuple>
#include <vector>

#include "../../utils/cli_fields.h"

namespace arenai::agent {

    struct ImpalaPpoHyperParams {
        float actor_learning_rate = 3e-4f;
        float critic_learning_rate = 3e-4f;
        int hidden_size_sensors = 64;
        std::vector<int> actor_hidden_sizes = {256, 128};
        std::vector<int> critic_hidden_sizes = {256, 128};
        std::vector<std::tuple<int, int>> vision_channels =
            {{3, 16}, {16, 32}, {32, 64}, {64, 128}};
        float initial_sigma = 0.5f;
        float initial_fire_proba = 0.5f;
        float initial_zoom_proba = 0.5f;
        int metric_window_size = 256;
        float gamma = 0.994f;
        float gae_lambda = 0.99f;
        float clip_epsilon = 0.2f;
        float target_kl = 0.05f;
        float grad_norm_max = 0.5f;
        float continuous_entropy_factor = 0.001f;
        float discrete_entropy_factor = 0.f;
        int actor_epochs = 8;
        int critic_epochs = 16;
        int rollout_size = 30 * 30;
        int minibatch_size = 128;
    };

    std::vector<CliField<ImpalaPpoHyperParams>> impala_ppo_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_IMPALA_PPO_HYPERPARAMS_H
