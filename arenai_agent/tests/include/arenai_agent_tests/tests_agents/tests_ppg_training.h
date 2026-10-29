//
// Created by claude on 29/10/2026.
//

#ifndef ARENAI_TESTS_PPG_TRAINING_H
#define ARENAI_TESTS_PPG_TRAINING_H

#include <memory>

#include <agents/ppg/ppg_factory.h>
#include <gtest/gtest.h>

// test code: the arenai sub-namespaces are used unqualified
using namespace arenai;

struct PpgTrainingTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

// triad built by hand to keep a handle on the actor's parameters
struct PpgTriad {
    std::shared_ptr<arenai::agent::DistillationActor> actor;
    std::shared_ptr<arenai::agent::PpgRolloutBuffer> rollout_buffer;
    std::shared_ptr<arenai::agent::PpgStepCollector> collector;
    std::shared_ptr<arenai::agent::TorchPpgAgent> agent;
    std::shared_ptr<arenai::agent::PpgTrainer> trainer;
};

class PpgTrainingTest : public testing::Test {
protected:
    // small enough for the trainer to trigger during the test loop
    static constexpr int ROLLOUT_SIZE = 3;
    // smaller than the number of valid rows so the loop exercises several minibatches
    static constexpr int MINIBATCH_SIZE = 4;

    torch::Device device{torch::kCPU};

    std::unique_ptr<arenai::agent::PpgTorchAgentFactory>
    make_factory(const PpgTrainingTestConfig &cfg) const;

    static arenai::agent::TorchState make_state(const PpgTrainingTestConfig &cfg, int nb_tanks);

    // target_kl = 0 : early stop disabled so every optimizer batch applies its update
    PpgTriad make_triad(const PpgTrainingTestConfig &cfg) const;

    // env loop: act -> transition -> maybe train
    static void run_env_loop(
        const PpgTriad &triad, const PpgTrainingTestConfig &cfg, int nb_steps, bool fire_possible);
};

#endif//ARENAI_TESTS_PPG_TRAINING_H
