//
// Created by claude on 01/07/2026.
//

#include <agents/sac/replay_buffer.h>

#include <arenai_train_tests/tests_replay_buffer/tests_replay_buffer_edge.h>

#include "./create_random_step.h"

using namespace arenai;
using namespace arenai::train;

TEST_F(ReplayBufferEdgeTest, SampleFromEmptyBufferDoesNotCrash) {
    SacReplayBuffer buffer(10);

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

TEST_F(ReplayBufferEdgeTest, RewardUnchangedAtSample) {
    SacReplayBuffer buffer(10);

    SacInputStep step;
    step.state.vision = torch::randint(255, {1, 3, 8, 8}, torch::kUInt8);
    step.state.proprioception = torch::randn({1, 5});
    step.action.continuous_action = torch::randn({1, 3});
    step.action.discrete_action = torch::zeros({1, 2});
    step.action.discrete_action[0][0] = 1.0f;
    step.reward = torch::full({1, 1}, 2.0f);
    step.done = torch::zeros({1, 1});
    step.truncated = torch::zeros({1, 1});

    buffer.add(step);
    buffer.add(step);

    const auto output = buffer.sample(1, torch::kCPU);

    ASSERT_NEAR(output.reward.item<float>(), 2.0f, 1e-5f)
        << "Base ReplayBuffer should return the stored reward unchanged at sample time";
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
