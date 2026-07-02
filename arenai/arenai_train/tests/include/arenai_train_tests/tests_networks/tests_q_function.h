//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_Q_FUNCTION_H
#define ARENAI_TESTS_Q_FUNCTION_H

#include <gtest/gtest.h>

typedef std::vector<int> HiddenLayers;
typedef uint32_t ContinuousActionsNb;
typedef uint32_t DiscreteActionsNb;

typedef uint32_t SensorsNb;
typedef uint32_t SensorsHiddenSize;

typedef uint32_t ActionsHiddenSize;

typedef uint32_t BatchSize;

class QFunctionTestParam : public testing::TestWithParam<std::tuple<
                               HiddenLayers, ContinuousActionsNb, DiscreteActionsNb, SensorsNb,
                               SensorsHiddenSize, ActionsHiddenSize, BatchSize>> {};

#endif//ARENAI_TESTS_Q_FUNCTION_H
