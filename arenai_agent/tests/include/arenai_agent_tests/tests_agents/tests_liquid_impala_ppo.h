//
// Created by claude on 27/10/2026.
//

#ifndef ARENAI_TESTS_LIQUID_IMPALA_PPO_H
#define ARENAI_TESTS_LIQUID_IMPALA_PPO_H

#include <filesystem>
#include <memory>

#include <agents/ppo_liquid_impala/liquid_impala_ppo_factory.h>
#include <gtest/gtest.h>

// test code: the arenai sub-namespaces are used unqualified
using namespace arenai;

struct LiquidImpalaPpoTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

class LiquidImpalaPpoAgentTest : public testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    std::unique_ptr<arenai::agent::LiquidImpalaPpoTorchAgentFactory>
    make_factory(const LiquidImpalaPpoTestConfig &cfg) const;

    static arenai::agent::TorchState make_state(const LiquidImpalaPpoTestConfig &cfg, int batch);

    std::filesystem::path tmp_dir;
    torch::Device device{torch::kCPU};
};

typedef LiquidImpalaPpoTestConfig LiquidImpalaPpoActShapeParam;

class LiquidImpalaPpoActShapeParamTest
    : public LiquidImpalaPpoAgentTest,
      public testing::WithParamInterface<LiquidImpalaPpoActShapeParam> {};

class LiquidImpalaPpoSaveLoadParamTest
    : public LiquidImpalaPpoAgentTest,
      public testing::WithParamInterface<LiquidImpalaPpoActShapeParam> {};

#endif//ARENAI_TESTS_LIQUID_IMPALA_PPO_H
