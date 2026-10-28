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
        float actor_learning_rate = 3e-4f;
        float critic_learning_rate = 3e-4f;
        int hidden_size_sensors = 64;
        std::vector<std::tuple<int, int>> vision_channels =
            {{3, 16}, {16, 32}, {32, 64}, {64, 128}};
        std::vector<int> group_norm_nums = {2, 4, 8, 16};
        int neuron_number = 128;
        int unfolding_steps = 6;
        float delta_t = 1.f / 10.f;
        int chunk_size = 30;
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
        int critic_epochs = 8;
        int rollout_size = 30 * 10;
        // one optimizer step per batch_size rows, accumulated over gpu_batch_size-row
        // forward / backward passes (the device memory bound)
        int actor_batch_size = 512;
        int actor_gpu_batch_size = 128;
        int critic_batch_size = 512;
        int critic_gpu_batch_size = 128;
    };

    std::vector<CliField<LiquidPpoHyperParams>> liquid_ppo_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_LIQUID_PPO_HYPERPARAMS_H
