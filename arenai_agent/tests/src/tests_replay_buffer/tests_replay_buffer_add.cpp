//
// Created by samuel on 30/06/2026.
//

#include <agents/sac/replay_buffer.h>

#include <arenai_agent_tests/tests_replay_buffer/tests_replay_buffer_add.h>

#include "./create_random_step.h"

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Normal: steps_to_add <= memory_size
// ========================================================================

TEST_P(ReplayBufferAddNormalTestParam, SizeMatchesAdded) {
    const auto [memory_size, to_add] = GetParam();

    SacReplayBuffer buffer(static_cast<int>(memory_size));

    for (uint32_t i = 0; i < to_add; ++i) buffer.add(create_random_step(8, 8, 3, 2, 5, false));

    ASSERT_EQ(buffer.size(), to_add);
}

INSTANTIATE_TEST_SUITE_P(
    ReplayBufferAddNormal, ReplayBufferAddNormalTestParam,
    testing::Combine(testing::Values(8, 16, 32), testing::Values(1, 2, 4, 8)));

// ========================================================================
// Overflow: steps_to_add > memory_size
// ========================================================================

TEST_P(ReplayBufferAddOverflowTestParam, SizeCappedAtMemorySize) {
    const auto [memory_size, to_add] = GetParam();

    SacReplayBuffer buffer(static_cast<int>(memory_size));

    for (uint32_t i = 0; i < to_add; ++i) buffer.add(create_random_step(8, 8, 3, 2, 5, false));

    ASSERT_EQ(buffer.size(), memory_size);
    ASSERT_LT(buffer.size(), to_add);
}

INSTANTIATE_TEST_SUITE_P(
    ReplayBufferAddOverflow, ReplayBufferAddOverflowTestParam,
    testing::Combine(testing::Values(2, 4, 8), testing::Values(10, 16, 32)));
