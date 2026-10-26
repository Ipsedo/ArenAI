//
// Created by claude on 26/10/2026.
//

#ifndef ARENAI_TESTS_IMPALA_PPO_H
#define ARENAI_TESTS_IMPALA_PPO_H

#include <filesystem>
#include <memory>

#include <agents/ppo_impala/impala_ppo_factory.h>
#include <gtest/gtest.h>

// test code: the arenai sub-namespaces are used unqualified
using namespace arenai;

struct ImpalaPpoTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

class ImpalaPpoAgentTest : public testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    std::unique_ptr<arenai::agent::ImpalaPpoTorchAgentFactory>
    make_factory(const ImpalaPpoTestConfig &cfg) const;

    static arenai::agent::TorchState make_state(const ImpalaPpoTestConfig &cfg, int batch);

    std::filesystem::path tmp_dir;
    torch::Device device{torch::kCPU};
};

typedef ImpalaPpoTestConfig ImpalaPpoActShapeParam;

class ImpalaPpoActShapeParamTest : public ImpalaPpoAgentTest,
                                   public testing::WithParamInterface<ImpalaPpoActShapeParam> {};

class ImpalaPpoSaveLoadParamTest : public ImpalaPpoAgentTest,
                                   public testing::WithParamInterface<ImpalaPpoActShapeParam> {};

#endif//ARENAI_TESTS_IMPALA_PPO_H
