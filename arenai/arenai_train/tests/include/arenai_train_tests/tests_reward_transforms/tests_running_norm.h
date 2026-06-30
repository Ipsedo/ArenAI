//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_RUNNING_NORM_H
#define ARENAI_TESTS_RUNNING_NORM_H

#include <gtest/gtest.h>

typedef int MemorySize;
typedef float RewardScale;

class NormalizedRewardTransformTest : public testing::Test {};

class NormalizedRewardTransformParamTest
    : public testing::TestWithParam<std::tuple<MemorySize, RewardScale>> {};

class NormalizedNonZeroTransformTest : public testing::Test {};

class NormalizedNonZeroTransformParamTest : public testing::TestWithParam<MemorySize> {};

#endif//ARENAI_TESTS_RUNNING_NORM_H
