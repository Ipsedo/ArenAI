//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_MISC_H
#define ARENAI_TESTS_MISC_H

#include <gtest/gtest.h>

typedef std::vector<int64_t> Shape;
typedef float LowerBound;
typedef float UpperBound;

class ClampModuleParamTest
    : public testing::TestWithParam<std::tuple<LowerBound, UpperBound, Shape>> {};

class ExpModuleParamTest : public testing::TestWithParam<Shape> {};

#endif//ARENAI_TESTS_MISC_H
