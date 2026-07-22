//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_TESTS_PPO_H
#define ARENAI_TESTS_PPO_H

#include <filesystem>
#include <memory>

#include <agents/ppo/ppo_factory.h>
#include <gtest/gtest.h>

struct PpoTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

class PpoAgentTest : public testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    std::unique_ptr<arenai::agent::PpoTorchAgentFactory>
    make_factory(const PpoTestConfig &cfg) const;

    static arenai::agent::TorchState make_state(const PpoTestConfig &cfg, int batch);

    std::filesystem::path tmp_dir;
    torch::Device device{torch::kCPU};
};

typedef PpoTestConfig PpoActShapeParam;

class PpoActShapeParamTest : public PpoAgentTest,
                             public testing::WithParamInterface<PpoActShapeParam> {};

class PpoSaveLoadParamTest : public PpoAgentTest,
                             public testing::WithParamInterface<PpoActShapeParam> {};

#endif//ARENAI_TESTS_PPO_H
