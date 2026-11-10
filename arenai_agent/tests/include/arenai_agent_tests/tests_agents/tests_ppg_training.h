#ifndef ARENAI_TESTS_PPG_TRAINING_H
#define ARENAI_TESTS_PPG_TRAINING_H

#include <memory>

#include <agents/ppg/ppg_factory.h>
#include <gtest/gtest.h>

using namespace arenai;

struct PpgTrainingTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

struct PpgTriad {
    std::shared_ptr<arenai::agent::DistillationActor> actor;
    std::shared_ptr<arenai::agent::PpgRolloutBuffer> rollout_buffer;
    std::shared_ptr<arenai::agent::PpgStepCollector> collector;
    std::shared_ptr<arenai::agent::TorchPpgAgent> agent;
    std::shared_ptr<arenai::agent::PpgTrainer> trainer;
};

class PpgTrainingTest : public testing::Test {
protected:
    static constexpr int ROLLOUT_SIZE = 3;
    static constexpr int MINIBATCH_SIZE = 4;

    torch::Device device{torch::kCPU};

    std::unique_ptr<arenai::agent::PpgTorchAgentFactory>
    make_factory(const PpgTrainingTestConfig &cfg) const;

    static arenai::agent::TorchState make_state(const PpgTrainingTestConfig &cfg, int nb_tanks);

    PpgTriad make_triad(const PpgTrainingTestConfig &cfg, int n_pi = 1) const;

    static void run_env_loop(
        const PpgTriad &triad, const PpgTrainingTestConfig &cfg, int nb_steps, bool fire_possible);
};

#endif//ARENAI_TESTS_PPG_TRAINING_H
