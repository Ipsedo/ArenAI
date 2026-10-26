//
// Created by claude on 26/10/2026.
//

#ifndef ARENAI_TESTS_IMPALA_PPO_TRAINING_H
#define ARENAI_TESTS_IMPALA_PPO_TRAINING_H

#include <memory>

#include <agents/ppo_impala/impala_ppo_factory.h>
#include <gtest/gtest.h>

// test code: the arenai sub-namespaces are used unqualified
using namespace arenai;

struct ImpalaPpoTrainingTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

class ImpalaPpoTrainingTest : public testing::Test {
protected:
    // small enough for the trainer to trigger during the test loop
    static constexpr int ROLLOUT_SIZE = 3;
    // smaller than the number of valid rows so the loop exercises several minibatches
    static constexpr int MINIBATCH_SIZE = 4;

    torch::Device device{torch::kCPU};

    std::unique_ptr<arenai::agent::ImpalaPpoTorchAgentFactory>
    make_factory(const ImpalaPpoTrainingTestConfig &cfg) const;

    static arenai::agent::TorchState
    make_state(const ImpalaPpoTrainingTestConfig &cfg, int nb_tanks);
};

#endif//ARENAI_TESTS_IMPALA_PPO_TRAINING_H
