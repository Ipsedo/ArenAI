//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_TRAIN_H
#define ARENAI_TRAIN_HOST_TRAIN_H

#include <filesystem>
#include <vector>

struct ModelOptions {
    std::vector<std::tuple<int, int>> vision_channels;
    std::vector<int> group_norm_nums;
    int hidden_size_sensors;
    int hidden_size_actions;
    std::vector<int> actor_hidden_size;
    std::vector<int> critic_hidden_size;
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
    float wanted_frequency;
    int nb_tanks;
    float initial_spawn_width;
    float initial_spawn_height;
    float final_spawn_width;
    float final_spawn_height;
};

float compute_potential_reward_scale(
    float wanted_frequency, float distance_scale, float target_reward = 0.25f,
    float typical_fraction = 0.1f);

void train_main(
    const EnvironmentOptions &environment_options, const ModelOptions &model_options,
    const TrainOptions &train_options);

#endif// ARENAI_TRAIN_HOST_TRAIN_H
