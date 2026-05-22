//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_TRAIN_H
#define ARENAI_TRAIN_HOST_TRAIN_H

#include <filesystem>
#include <vector>

#include <arenai_train/metric.h>

struct ModelOptions {
    std::vector<std::tuple<int, int>> vision_channels;
    std::vector<int> group_norm_nums;
    int hidden_size_sensors;
    int hidden_size_actions;
    int actor_hidden_size;
    int critic_hidden_size;
    float tau;
    float gamma;
    float initial_alpha_continuous;
    float initial_alpha_discrete;
};

struct TrainOptions {
    std::filesystem::path output_folder;
    std::filesystem::path android_asset_folder;
    float actor_learning_rate;
    float critic_learning_rate;
    float alpha_learning_rate;
    float potential_reward_scale;
    int epochs;
    int batch_size;
    int max_episode_steps;
    int nb_episodes;
    int replay_buffer_size;
    int train_every;
    int save_every;
    bool cuda;
    int metric_window_size;
};

struct EnvironmentOptions {
    int nb_tanks;
    float initial_spawn_width;
    float initial_spawn_height;
    float final_spawn_width;
    float final_spawn_height;
};

void train_main(
    float wanted_frequency, const EnvironmentOptions &environment_options,
    const ModelOptions &model_options, const TrainOptions &train_options);

#endif// ARENAI_TRAIN_HOST_TRAIN_H
