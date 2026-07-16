//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_TRUNCATED_NORMAL_H
#define ARENAI_TESTS_TRUNCATED_NORMAL_H

#include <gtest/gtest.h>

typedef int UpperBound;
typedef int LowerBound;

typedef std::vector<int64_t> Shape;

class TruncatedNormalTestParam
    : public testing::TestWithParam<std::tuple<UpperBound, LowerBound, Shape>> {};

#endif//ARENAI_TESTS_TRUNCATED_NORMAL_H
