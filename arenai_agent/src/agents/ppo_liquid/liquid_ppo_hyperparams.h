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
        float critic_learning_rate = 1e-4f;
        int hidden_size_sensors = 64;
        std::vector<std::tuple<int, int>> vision_channels =
            {{3, 16}, {16, 32}, {32, 64}, {64, 128}};
        std::vector<int> group_norm_nums = {2, 4, 8, 16};
        int neuron_number = 128;
        int unfolding_steps = 6;
        // seconds between two agent decisions (action repeat 3 at 30 Hz)
        float delta_t = 1.f / 10.f;
        int chunk_size = 30;
        float initial_sigma = 0.5f;
        float initial_fire_proba = 0.1f;
        float initial_zoom_proba = 0.25f;
        int metric_window_size = 256;
        // per agent decision (action repeat 3 at 30 Hz): 0.997^3, same ~11 s horizon
        float gamma = 0.991f;
        float gae_lambda = 0.97f;
        float clip_epsilon = 0.2f;
        float target_kl = 0.05f;
        float grad_norm_max = 0.5f;
        std::vector<float> continuous_entropy_factor = {0.001f, 0.001f, 0.001f, 0.001f};
        std::vector<float> discrete_entropy_factor = {0.001f, 0.001f};
        int epochs = 2;
        int rollout_size = 30 * 30;
        int minibatch_size = 1536;
    };

    std::vector<CliField<LiquidPpoHyperParams>> liquid_ppo_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_LIQUID_PPO_HYPERPARAMS_H
