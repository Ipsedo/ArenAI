//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_TESTS_PPO_ROLLOUT_BUFFER_H
#define ARENAI_TESTS_PPO_ROLLOUT_BUFFER_H

#include <agents/ppo/ppo_rollout_buffer.h>
#include <gtest/gtest.h>

class PpoRolloutBufferTest : public testing::Test {
protected:
    static constexpr int NB_TANKS = 2;
    static constexpr int VISION_SIZE = 4;
    static constexpr int NB_SENSORS = 3;
    static constexpr int NB_CONTINUOUS_ACTIONS = 2;
    static constexpr int NB_DISCRETE_ACTIONS = 2;

    static arenai::agent::TorchState make_state();

    static arenai::agent::PpoInputStep
    make_step(const arenai::agent::TorchState &state, const torch::Tensor &done);

    static arenai::agent::PpoInputStep make_step(const arenai::agent::TorchState &state);
};

#endif//ARENAI_TESTS_PPO_ROLLOUT_BUFFER_H
