//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_TESTS_LIQUID_PPO_ROLLOUT_BUFFER_H
#define ARENAI_TESTS_LIQUID_PPO_ROLLOUT_BUFFER_H

#include <agents/ppo_liquid/liquid_ppo_rollout_buffer.h>
#include <gtest/gtest.h>

class LiquidPpoRolloutBufferTest : public testing::Test {
protected:
    static constexpr int NB_TANKS = 2;
    static constexpr int VISION_SIZE = 4;
    static constexpr int NB_SENSORS = 3;
    static constexpr int NB_CONTINUOUS_ACTIONS = 2;
    static constexpr int NB_DISCRETE_ACTIONS = 2;
    static constexpr int NEURON_NUMBER = 4;

    static arenai::agent::TorchState make_state();

    static arenai::agent::LiquidPpoInputStep make_step(
        const arenai::agent::TorchState &state, const torch::Tensor &done, bool episode_start);

    static arenai::agent::LiquidPpoInputStep make_step(const arenai::agent::TorchState &state);
};

#endif//ARENAI_TESTS_LIQUID_PPO_ROLLOUT_BUFFER_H
