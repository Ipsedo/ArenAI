#ifndef ARENAI_SAC_HYPERPARAMS_H
#define ARENAI_SAC_HYPERPARAMS_H

#include <tuple>
#include <vector>

#include "../../utils/cli_fields.h"

namespace arenai::agent {

    struct SacHyperParams {
        float actor_learning_rate = 3e-4f;
        float critic_learning_rate = 3e-4f;
        float alpha_learning_rate = 3e-5f;
        int hidden_size_sensors = 64;
        std::vector<int> actor_hidden_sizes = {256, 128};
        std::vector<int> critic_hidden_sizes = {256, 128};
        std::vector<std::tuple<int, int>> vision_channels =
            {{3, 16}, {16, 32}, {32, 64}, {64, 128}};
        std::vector<int> group_norm_nums = {2, 4, 8, 16};
        float initial_sigma = 0.3f;
        float initial_fire_proba = 0.5f;
        float initial_zoom_proba = 0.5f;
        int metric_window_size = 256;
        float gamma = 0.99f;
        float tau = 0.005f;
        float initial_alpha = 5e-4f;
        float target_sigma = 0.1f;
        float discrete_target_entropy_ratio = 0.3f;
        int buffer_size = 200000;
        float armed_fraction = 0.7f;
        int learning_starts = 5000;
        int gradient_steps = 2;
        int batch_size = 512;
        int gpu_batch_size = 1024;
    };

    std::vector<CliField<SacHyperParams>> sac_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_SAC_HYPERPARAMS_H
