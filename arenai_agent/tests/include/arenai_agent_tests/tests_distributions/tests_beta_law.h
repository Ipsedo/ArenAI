//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_BETA_LAW_H
#define ARENAI_TESTS_BETA_LAW_H

#include <gtest/gtest.h>

typedef std::vector<int64_t> Shape;

class BetaLawTest : public testing::Test {};
class BetaLawParamTest : public testing::TestWithParam<Shape> {};

#endif//ARENAI_TESTS_BETA_LAW_H
