//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_MULTINOMIAL_H
#define ARENAI_TESTS_MULTINOMIAL_H

#include <gtest/gtest.h>

typedef int NbActions;
typedef float ShootProbability;

class MultinomialTest : public testing::Test {};
class MultinomialShapeParamTest
    : public testing::TestWithParam<std::tuple<int /*batch_size*/, NbActions>> {};
class MultinomialMaxEntropyParamTest : public testing::TestWithParam<NbActions> {};
class MultinomialTargetEntropyParamTest : public testing::TestWithParam<ShootProbability> {};

#endif//ARENAI_TESTS_MULTINOMIAL_H
