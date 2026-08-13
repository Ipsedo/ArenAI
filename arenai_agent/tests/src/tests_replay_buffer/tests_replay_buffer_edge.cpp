//
// Created by claude on 01/07/2026.
//

#include <cmath>

#include <agents/sac/sac_replay_buffer.h>

#include <arenai_agent_tests/tests_replay_buffer/tests_replay_buffer_edge.h>

#include "./create_random_step.h"

using namespace arenai;
using namespace arenai::agent;

TEST_F(ReplayBufferEdgeTest, SampleFromEmptyBufferDoesNotCrash) {
    const SacReplayBuffer buffer(10);

    ASSERT_EQ(buffer.size(), 0u);

    // Sampling from empty buffer should either throw or return a valid (if degenerate) result
    // but must NOT produce undefined behavior
    ASSERT_ANY_THROW(buffer.sample(1, torch::kCPU))
        << "Sampling from empty buffer should throw (randint with upper_bound=0 is invalid)";
}

TEST_F(ReplayBufferEdgeTest, SampleFromSingleElement) {
    SacReplayBuffer buffer(10);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));
    buffer.finish_episode(create_random_state(8, 8, 5));

    ASSERT_EQ(buffer.size(), 2u);

    const auto output = buffer.sample(1, torch::kCPU);

    ASSERT_EQ(output.state.vision.size(0), 1);
    ASSERT_EQ(output.action.continuous_action.size(0), 1);
}

TEST_F(ReplayBufferEdgeTest, SampleBatchLargerThanSingleElement) {
    SacReplayBuffer buffer(10);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));
    buffer.finish_episode(create_random_state(8, 8, 5));

    const auto output = buffer.sample(5, torch::kCPU);

    ASSERT_EQ(output.state.vision.size(0), 1)
        << "Batch size should be clamped to the single available transition";
}

namespace {
    SacInputStep create_step_with_reward(const float reward) {
        SacInputStep step;
        step.state.vision = torch::randint(255, {1, 3, 8, 8}, torch::kUInt8);
        step.state.proprioception = torch::randn({1, 5});
        step.action.continuous_action = torch::randn({1, 3});
        step.action.discrete_action = torch::zeros({1, 2});
        step.action.discrete_action[0][0] = 1.0f;
        step.reward = torch::full({1, 1}, reward);
        step.done = torch::zeros({1, 1});
        return step;
    }
}// namespace

TEST_F(ReplayBufferEdgeTest, ConstantRewardsAreNotRescaled) {
    SacReplayBuffer buffer(10);

    buffer.add(create_step_with_reward(2.0f));
    buffer.add(create_step_with_reward(2.0f));

    const auto output = buffer.sample(1, torch::kCPU);

    ASSERT_NEAR(output.reward.item<float>(), 2.0f, 1e-5f)
        << "Zero reward variance must keep the scale at 1 (no division by ~0)";
}

TEST_F(ReplayBufferEdgeTest, RewardDividedByRunningStdAtSample) {
    SacReplayBuffer buffer(10);

    // rewards {0, 4}: mean 2, population std 2
    buffer.add(create_step_with_reward(0.0f));
    buffer.add(create_step_with_reward(4.0f));
    buffer.finish_episode(create_random_state(8, 8, 5));

    const auto output = buffer.sample(64, torch::kCPU);

    const auto rewards = output.reward.reshape({-1});
    for (int64_t i = 0; i < rewards.size(0); i++) {
        const auto r = rewards[i].item<float>();
        ASSERT_TRUE(std::abs(r - 0.0f) < 1e-5f || std::abs(r - 2.0f) < 1e-5f)
            << "Sampled reward " << r << " should be a stored reward divided by the running std";
    }
}

TEST_F(ReplayBufferEdgeTest, DeadStepIsStoredAsTerminal) {
    SacReplayBuffer buffer(10);

    buffer.add(create_random_step(8, 8, 3, 2, 5, true));
    buffer.finish_episode(create_random_state(8, 8, 5));

    const auto output = buffer.sample(1, torch::kCPU);

    ASSERT_TRUE(output.done.to(torch::kBool).item<bool>())
        << "A termination (done) must sample done=true";
}

TEST_F(ReplayBufferEdgeTest, SampleWithZeroBatchSize) {
    SacReplayBuffer buffer(10);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));
    buffer.add(create_random_step(8, 8, 3, 2, 5, false));

    const auto output = buffer.sample(0, torch::kCPU);

    ASSERT_EQ(output.state.vision.size(0), 1) << "Batch size 0 should be clamped to 1";
}

TEST_F(ReplayBufferEdgeTest, SampleWithNegativeBatchSize) {
    SacReplayBuffer buffer(10);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));
    buffer.finish_episode(create_random_state(8, 8, 5));

    const auto output = buffer.sample(-5, torch::kCPU);

    ASSERT_EQ(output.state.vision.size(0), 1) << "Negative batch size should be clamped to 1";
}

TEST_F(ReplayBufferEdgeTest, CircularOverwriteKeepsMaxSize) {
    SacReplayBuffer buffer(3);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));
    buffer.add(create_random_step(8, 8, 3, 2, 5, false));
    ASSERT_EQ(buffer.size(), 2u);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));
    ASSERT_EQ(buffer.size(), 3u);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));
    ASSERT_EQ(buffer.size(), 3u) << "Buffer size should not exceed capacity after wraparound";

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));
    ASSERT_EQ(buffer.size(), 3u);
}
