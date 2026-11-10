#ifndef ARENAI_TESTS_SAC_TRAINING_H
#define ARENAI_TESTS_SAC_TRAINING_H

#include <memory>

#include <agents/sac/sac_factory.h>
#include <gtest/gtest.h>

// test code: the arenai sub-namespaces are used unqualified
using namespace arenai;

struct SacTrainingTestConfig {
    int vision_height;
    int vision_width;
    int nb_sensors;
    int nb_continuous_actions;
    int nb_discrete_actions;
};

struct SacTrainingTriad {
    std::shared_ptr<arenai::agent::Actor> actor;
    std::shared_ptr<arenai::agent::SacReplayBuffer> replay_buffer;
    std::shared_ptr<arenai::agent::SacStepCollector> collector;
    std::shared_ptr<arenai::agent::TorchSacAgent> agent;
    std::shared_ptr<arenai::agent::SacTrainer> trainer;
};

class SacTrainingTest : public testing::Test {
protected:
    static constexpr int NB_TANKS = 2;
    // transitions before the first gradient step
    static constexpr int LEARNING_STARTS = 4;
    // optimizer batch, cut into two device blocks
    static constexpr int BATCH_SIZE = 8;
    static constexpr int GPU_BATCH_SIZE = 4;

    torch::Device device{torch::kCPU};

    // the triad built by hand, to keep a handle on the actor's parameters
    SacTrainingTriad make_triad(const SacTrainingTestConfig &cfg) const;

    static arenai::agent::TorchState make_state(const SacTrainingTestConfig &cfg, int nb_tanks);
};

#endif//ARENAI_TESTS_SAC_TRAINING_H
