//
// Created by claude on 01/07/2026.
//

#ifndef ARENAI_TESTS_SAC_TRAINING_H
#define ARENAI_TESTS_SAC_TRAINING_H

#include <filesystem>
#include <memory>

#include <agents/sac/sac_factory.h>
#include <gtest/gtest.h>

struct SacTrainingTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

class SacTrainingTest : public testing::Test {
protected:
    torch::Device device{torch::kCPU};

    std::unique_ptr<arenai::agent::SacTorchAgentFactory>
    make_factory(const SacTrainingTestConfig &cfg) const;

    static arenai::agent::TorchState make_state(const SacTrainingTestConfig &cfg);
};

#endif//ARENAI_TESTS_SAC_TRAINING_H
