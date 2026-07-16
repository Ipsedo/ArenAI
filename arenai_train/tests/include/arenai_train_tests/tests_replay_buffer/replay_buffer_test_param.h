//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_REPLAY_BUFFER_TEST_PARAM_H
#define ARENAI_REPLAY_BUFFER_TEST_PARAM_H

#include <gtest/gtest.h>

typedef uint32_t MemorySize;

template<typename... OtherArgs>
class ReplayBufferTestParam : public testing::TestWithParam<std::tuple<MemorySize, OtherArgs...>> {
};

#endif//ARENAI_REPLAY_BUFFER_TEST_PARAM_H
