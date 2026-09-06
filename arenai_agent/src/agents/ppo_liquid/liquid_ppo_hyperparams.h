//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_LIQUID_PPO_HYPERPARAMS_H
#define ARENAI_LIQUID_PPO_HYPERPARAMS_H

#include <tuple>
#include <vector>

#include "../../utils/cli_fields.h"

namespace arenai::agent {

    // Member initializers are the CLI defaults (single source of truth).
    struct LiquidPpoHyperParams {
        float actor_learning_rate = 1e-4f;
        float critic_learning_rate = 3e-4f;
        int hidden_size_sensors = 128;
        std::vector<std::tuple<int, int>> vision_channels = {{3, 8},   {8, 16},  {16, 24},
                                                             {24, 32}, {32, 48}, {48, 64}};
        std::vector<int> group_norm_nums = {1, 2, 3, 4, 6, 8};
        int neuron_number = 128;
        int unfolding_steps = 6;
        float delta_t = 1.f / 30.f;
        // TBPTT window: length of the contiguous sequences the updates run on
        int chunk_size = 30;
        float initial_sigma = 0.5f;
        float initial_fire_proba = 0.4f;
        int metric_window_size = 256;
        float gamma = 0.997f;
        float gae_lambda = 0.99f;
        float clip_epsilon = 0.2f;
        float target_kl = 0.05f;
        float grad_norm_max = 0.5f;
        float continuous_target_entropy = 0.4f;
        float discrete_target_entropy_factor = 0.2f;
        int epochs = 2;
        int rollout_size = 30 * 30;
        // expressed in steps: a minibatch holds minibatch_size / chunk_size chunks
        int minibatch_size = 1024;
    };

    std::vector<CliField<LiquidPpoHyperParams>> liquid_ppo_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_LIQUID_PPO_HYPERPARAMS_H
