#ifndef ARENAI_TESTS_PPO_TRAINING_H
#define ARENAI_TESTS_PPO_TRAINING_H

#include <memory>

#include <agents/ppo/ppo_factory.h>
#include <gtest/gtest.h>

using namespace arenai;

struct PpoTrainingTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

class PpoTrainingTest : public testing::Test {
protected:
    static constexpr int ROLLOUT_SIZE = 3;
    static constexpr int MINIBATCH_SIZE = 4;

    torch::Device device{torch::kCPU};

    std::unique_ptr<arenai::agent::PpoTorchAgentFactory>
    make_factory(const PpoTrainingTestConfig &cfg) const;

    static arenai::agent::TorchState make_state(const PpoTrainingTestConfig &cfg, int nb_tanks);
};

#endif//ARENAI_TESTS_PPO_TRAINING_H
