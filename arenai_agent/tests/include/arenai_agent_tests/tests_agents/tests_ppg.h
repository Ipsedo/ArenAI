//
// Created by claude on 29/10/2026.
//

#ifndef ARENAI_TESTS_PPG_H
#define ARENAI_TESTS_PPG_H

#include <filesystem>
#include <memory>

#include <agents/ppg/ppg_factory.h>
#include <gtest/gtest.h>

// test code: the arenai sub-namespaces are used unqualified
using namespace arenai;

struct PpgTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

class PpgAgentTest : public testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    std::unique_ptr<arenai::agent::PpgTorchAgentFactory>
    make_factory(const PpgTestConfig &cfg) const;

    static arenai::agent::TorchState make_state(const PpgTestConfig &cfg, int batch);

    std::filesystem::path tmp_dir;
    torch::Device device{torch::kCPU};
};

typedef PpgTestConfig PpgActShapeParam;

class PpgActShapeParamTest : public PpgAgentTest,
                             public testing::WithParamInterface<PpgActShapeParam> {};

class PpgSaveLoadParamTest : public PpgAgentTest,
                             public testing::WithParamInterface<PpgActShapeParam> {};

#endif//ARENAI_TESTS_PPG_H
