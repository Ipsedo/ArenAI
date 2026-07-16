//
// Created by samuel on 30/06/2026.
//

#include <arenai_train/replay_buffer.h>
#include <arenai_train_tests/tests_replay_buffer/tests_replay_buffer_sample.h>

#include "./create_random_step.h"

using namespace arenai;
using namespace arenai::train;

namespace {
    constexpr int WIDTH = 16, HEIGHT = 8;
    constexpr int CONT_ACTIONS_NB = 3, DISCRETE_ACTIONS_NB = 4;
    constexpr int SENSORS_NB = 5;

    void assert_sample_shapes(const TorchStep &output, const int expected_batch_size) {
        const auto &[state, action, reward, done, next_state] = output;

        ASSERT_EQ(state.vision.ndimension(), 4);
        ASSERT_EQ(state.vision.size(0), expected_batch_size);
        ASSERT_EQ(state.vision.size(1), 3);
        ASSERT_EQ(state.vision.size(2), HEIGHT);
        ASSERT_EQ(state.vision.size(3), WIDTH);

        ASSERT_EQ(state.proprioception.ndimension(), 2);
        ASSERT_EQ(state.proprioception.size(0), expected_batch_size);
        ASSERT_EQ(state.proprioception.size(1), SENSORS_NB);

        ASSERT_EQ(action.continuous_action.ndimension(), 2);
        ASSERT_EQ(action.continuous_action.size(0), expected_batch_size);
        ASSERT_EQ(action.continuous_action.size(1), CONT_ACTIONS_NB);

        ASSERT_EQ(action.discrete_action.ndimension(), 2);
        ASSERT_EQ(action.discrete_action.size(0), expected_batch_size);
        ASSERT_EQ(action.discrete_action.size(1), DISCRETE_ACTIONS_NB);

        ASSERT_EQ(reward.ndimension(), 2);
        ASSERT_EQ(reward.size(0), expected_batch_size);
        ASSERT_EQ(reward.size(1), 1);

        ASSERT_EQ(done.ndimension(), 2);
        ASSERT_EQ(done.size(0), expected_batch_size);
        ASSERT_EQ(done.size(1), 1);

        ASSERT_EQ(next_state.vision.ndimension(), 4);
        ASSERT_EQ(next_state.vision.size(0), expected_batch_size);
        ASSERT_EQ(next_state.vision.size(1), 3);
        ASSERT_EQ(next_state.vision.size(2), HEIGHT);
        ASSERT_EQ(next_state.vision.size(3), WIDTH);

        ASSERT_EQ(next_state.proprioception.ndimension(), 2);
        ASSERT_EQ(next_state.proprioception.size(0), expected_batch_size);
        ASSERT_EQ(next_state.proprioception.size(1), SENSORS_NB);
    }

    void fill_buffer(ReplayBuffer &buffer, const uint32_t count) {
        for (uint32_t i = 0; i < count; ++i)
            buffer.add(create_random_step(
                WIDTH, HEIGHT, CONT_ACTIONS_NB, DISCRETE_ACTIONS_NB, SENSORS_NB, false));
    }
}// namespace

// ========================================================================
// Normal batch size: batch_size <= steps_added <= memory_size
// ========================================================================

TEST_P(ReplayBufferSampleNormalTestParam, BatchSizeRespected) {
    const auto [memory_size, batch_size, steps_to_add] = GetParam();

    ReplayBuffer buffer(static_cast<int>(memory_size));
    fill_buffer(buffer, steps_to_add);

    const auto output = buffer.sample(static_cast<int>(batch_size), torch::kCPU);

    assert_sample_shapes(output, static_cast<int>(batch_size));
}

INSTANTIATE_TEST_SUITE_P(
    ReplayBufferSampleNormal, ReplayBufferSampleNormalTestParam,
    testing::Combine(
        testing::Values(16, 32), testing::Values(1, 4, 8), testing::Values(8, 16, 32)));

// ========================================================================
// Overflow batch size: batch_size > memory_size (buffer full)
// ========================================================================

TEST_P(ReplayBufferSampleOverflowTestParam, BatchSizeClampedToMemorySize) {
    const auto [memory_size, batch_size, steps_to_add] = GetParam();

    ReplayBuffer buffer(static_cast<int>(memory_size));
    fill_buffer(buffer, steps_to_add);

    const auto output = buffer.sample(static_cast<int>(batch_size), torch::kCPU);

    assert_sample_shapes(output, static_cast<int>(memory_size));
}

INSTANTIATE_TEST_SUITE_P(
    ReplayBufferSampleOverflow, ReplayBufferSampleOverflowTestParam,
    testing::Combine(
        testing::Values(4, 8), testing::Values(16, 32, 64), testing::Values(8, 16, 32)));

// ========================================================================
// Underflow batch size: batch_size > steps_added (buffer not full yet)
// ========================================================================

TEST_P(ReplayBufferSampleUnderflowTestParam, BatchSizeClampedToBufferSize) {
    const auto [memory_size, batch_size, steps_to_add] = GetParam();

    ReplayBuffer buffer(static_cast<int>(memory_size));
    fill_buffer(buffer, steps_to_add);

    const auto output = buffer.sample(static_cast<int>(batch_size), torch::kCPU);

    assert_sample_shapes(output, static_cast<int>(steps_to_add));
}

INSTANTIATE_TEST_SUITE_P(
    ReplayBufferSampleUnderflow, ReplayBufferSampleUnderflowTestParam,
    testing::Combine(testing::Values(32, 64), testing::Values(16, 32), testing::Values(1, 2, 4)));

// ========================================================================
// Double overflow: batch_size > memory_size > steps_added
// ========================================================================

TEST_P(ReplayBufferSampleDoubleOverflowTestParam, BatchSizeClampedToBufferSize) {
    const auto [memory_size, batch_size, steps_to_add] = GetParam();

    ReplayBuffer buffer(static_cast<int>(memory_size));
    fill_buffer(buffer, steps_to_add);

    const auto output = buffer.sample(static_cast<int>(batch_size), torch::kCPU);

    assert_sample_shapes(output, static_cast<int>(steps_to_add));
}

INSTANTIATE_TEST_SUITE_P(
    ReplayBufferSampleDoubleOverflow, ReplayBufferSampleDoubleOverflowTestParam,
    testing::Combine(testing::Values(8, 16), testing::Values(32, 64), testing::Values(1, 2, 4)));
