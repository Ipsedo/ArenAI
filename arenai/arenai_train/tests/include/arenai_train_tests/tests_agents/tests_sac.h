//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_SAC_H
#define ARENAI_TESTS_SAC_H

#include <filesystem>
#include <memory>

#include <agents/sac.h>
#include <gtest/gtest.h>

struct SacTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

class SacAgentTest : public testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    std::unique_ptr<SacAgent> make_agent(const SacTestConfig &cfg) const;

    static std::unique_ptr<ReplayBuffer> make_filled_buffer(const SacTestConfig &cfg, int n_steps);

    std::filesystem::path tmp_dir;
    torch::Device device{torch::kCPU};
};

typedef SacTestConfig ActShapeParam;

class SacActShapeParamTest : public SacAgentTest,
                             public testing::WithParamInterface<ActShapeParam> {};

class SacSaveLoadParamTest : public SacAgentTest,
                             public testing::WithParamInterface<ActShapeParam> {};

#endif//ARENAI_TESTS_SAC_H
