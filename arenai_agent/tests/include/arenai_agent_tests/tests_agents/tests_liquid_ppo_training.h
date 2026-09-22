//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_TESTS_LIQUID_PPO_TRAINING_H
#define ARENAI_TESTS_LIQUID_PPO_TRAINING_H

#include <memory>

#include <agents/ppo_liquid/liquid_ppo_factory.h>
#include <gtest/gtest.h>

struct LiquidPpoTrainingTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

class LiquidPpoTrainingTest : public testing::Test {
protected:
    // small enough for the trainer to trigger during the test loop
    static constexpr int ROLLOUT_SIZE = 4;
    // smaller than the number of valid rows so the loop exercises several minibatches
    static constexpr int MINIBATCH_SIZE = 4;
    static constexpr int CHUNK_SIZE = 2;
    static constexpr int NEURON_NUMBER = 16;
    static constexpr int UNFOLDING_STEPS = 2;
    static constexpr float DELTA_T = 1.f / 30.f;

    torch::Device device{torch::kCPU};

    std::unique_ptr<arenai::agent::LiquidPpoTorchAgentFactory>
    make_factory(const LiquidPpoTrainingTestConfig &cfg) const;

    static arenai::agent::TorchState
    make_state(const LiquidPpoTrainingTestConfig &cfg, int nb_tanks);
};

#endif//ARENAI_TESTS_LIQUID_PPO_TRAINING_H
