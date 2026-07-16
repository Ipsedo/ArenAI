//
// Created by claude on 01/07/2026.
//

#include <arenai_train/replay_buffer.h>
#include <arenai_train_tests/tests_replay_buffer/tests_replay_buffer_edge.h>

#include "./create_random_step.h"

using namespace arenai;
using namespace arenai::train;

TEST_F(ReplayBufferEdgeTest, SampleFromEmptyBufferDoesNotCrash) {
    ReplayBuffer buffer(10);

    ASSERT_EQ(buffer.size(), 0u);

    // Sampling from empty buffer should either throw or return a valid (if degenerate) result
    // but must NOT produce undefined behavior
    ASSERT_ANY_THROW(buffer.sample(1, torch::kCPU))
        << "Sampling from empty buffer should throw (randint with upper_bound=0 is invalid)";
}

TEST_F(ReplayBufferEdgeTest, SampleFromSingleElement) {
    ReplayBuffer buffer(10);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));

    ASSERT_EQ(buffer.size(), 1u);

    const auto output = buffer.sample(1, torch::kCPU);

    ASSERT_EQ(output.state.vision.size(0), 1);
    ASSERT_EQ(output.action.continuous_action.size(0), 1);
}

TEST_F(ReplayBufferEdgeTest, SampleBatchLargerThanSingleElement) {
    ReplayBuffer buffer(10);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));

    const auto output = buffer.sample(5, torch::kCPU);

    ASSERT_EQ(output.state.vision.size(0), 1) << "Batch size should be clamped to buffer size (1)";
}

TEST_F(ReplayBufferEdgeTest, RewardUnchangedAtSample) {
    ReplayBuffer buffer(10);

    TorchInputStep step;
    step.state.vision = torch::randint(255, {3, 8, 8}, torch::kUInt8);
    step.state.proprioception = torch::randn({5});
    step.action.continuous_action = torch::randn({3});
    step.action.discrete_action = torch::zeros({2});
    step.action.discrete_action[0] = 1.0f;
    step.main_reward = torch::tensor({2.0f});
    step.potential_reward = torch::tensor({0.0f});
    step.done = torch::tensor({0.0f});
    step.next_state.vision = torch::randint(255, {3, 8, 8}, torch::kUInt8);
    step.next_state.proprioception = torch::randn({5});

    buffer.add(step);

    const auto output = buffer.sample(1, torch::kCPU);

    ASSERT_NEAR(output.reward.item<float>(), 2.0f, 1e-5f)
        << "Base ReplayBuffer should return the stored reward unchanged at sample time";
}

TEST_F(ReplayBufferEdgeTest, SampleWithZeroBatchSize) {
    ReplayBuffer buffer(10);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));
    buffer.add(create_random_step(8, 8, 3, 2, 5, false));

    const auto output = buffer.sample(0, torch::kCPU);

    ASSERT_EQ(output.state.vision.size(0), 1) << "Batch size 0 should be clamped to 1";
}

TEST_F(ReplayBufferEdgeTest, SampleWithNegativeBatchSize) {
    ReplayBuffer buffer(10);

    buffer.add(create_random_step(8, 8, 3, 2, 5, false));

    const auto output = buffer.sample(-5, torch::kCPU);

    ASSERT_EQ(output.state.vision.size(0), 1) << "Negative batch size should be clamped to 1";
}

TEST_F(ReplayBufferEdgeTest, CircularOverwriteKeepsMaxSize) {
    ReplayBuffer buffer(3);

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
