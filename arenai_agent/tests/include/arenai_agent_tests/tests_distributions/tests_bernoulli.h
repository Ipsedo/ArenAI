//
// Created by samuel on 18/09/2026.
//

#ifndef ARENAI_TESTS_BERNOULLI_H
#define ARENAI_TESTS_BERNOULLI_H

#include <gtest/gtest.h>

typedef int NbActions;

class BernoulliTest : public testing::Test {};
class BernoulliShapeParamTest
    : public testing::TestWithParam<std::tuple<int /*batch_size*/, NbActions>> {};

#endif//ARENAI_TESTS_BERNOULLI_H
