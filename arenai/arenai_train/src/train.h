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
    int nb_tanks;
    std::filesystem::path output_folder;
    std::filesystem::path android_asset_folder;
    float learning_rate;
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

void train_main(
    float wanted_frequency, const ModelOptions &model_options, const TrainOptions &train_options);

#endif// ARENAI_TRAIN_HOST_TRAIN_H
