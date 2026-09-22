//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_TESTS_LIQUID_PPO_H
#define ARENAI_TESTS_LIQUID_PPO_H

#include <filesystem>
#include <memory>

#include <agents/ppo_liquid/liquid_ppo_factory.h>
#include <gtest/gtest.h>

struct LiquidPpoTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

class LiquidPpoAgentTest : public testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    std::unique_ptr<arenai::agent::LiquidPpoTorchAgentFactory>
    make_factory(const LiquidPpoTestConfig &cfg) const;

    static arenai::agent::TorchState make_state(const LiquidPpoTestConfig &cfg, int batch);

    std::filesystem::path tmp_dir;
    torch::Device device{torch::kCPU};
};

typedef LiquidPpoTestConfig LiquidPpoActShapeParam;

class LiquidPpoActShapeParamTest : public LiquidPpoAgentTest,
                                   public testing::WithParamInterface<LiquidPpoActShapeParam> {};

class LiquidPpoSaveLoadParamTest : public LiquidPpoAgentTest,
                                   public testing::WithParamInterface<LiquidPpoActShapeParam> {};

#endif//ARENAI_TESTS_LIQUID_PPO_H
