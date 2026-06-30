//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_GAUSSIAN_TANH_H
#define ARENAI_TESTS_GAUSSIAN_TANH_H

#include <gtest/gtest.h>

typedef std::vector<int64_t> Shape;
typedef float TargetSigma;

class GaussianTanhTest : public testing::Test {};
class GaussianTanhShapeParamTest : public testing::TestWithParam<Shape> {};
class GaussianTanhTargetEntropyParamTest
    : public testing::TestWithParam<std::tuple<int /*nb_actions*/, TargetSigma>> {};

#endif//ARENAI_TESTS_GAUSSIAN_TANH_H
