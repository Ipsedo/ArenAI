//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_AGENT_HOST_TRAIN_H
#define ARENAI_AGENT_HOST_TRAIN_H

#include <filesystem>
#include <memory>

#include "./agents/torch_factory.h"

namespace arenai::agent {

    struct TrainOptions {
        std::filesystem::path output_folder;
        std::filesystem::path resources_folder;
        int max_episode_steps;
        int nb_episodes;
        int save_every;
        bool cuda;
    };

    struct EnvironmentOptions {
        float wanted_frequency;
        int nb_tanks;
        int vision_height;
        int vision_width;
        float initial_spawn_width;
        float initial_spawn_height;
        float final_spawn_width;
        float final_spawn_height;
        int num_threads;
    };

    void train_main(
        const EnvironmentOptions &environment_options, const TrainOptions &train_options,
        const std::unique_ptr<AbstractTorchAgentFactory> &agent_factory);

}// namespace arenai::agent

#endif// ARENAI_AGENT_HOST_TRAIN_H
