//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_Q_FUNCTION_H
#define ARENAI_TESTS_Q_FUNCTION_H

#include <gtest/gtest.h>

typedef std::vector<int> HiddenLayers;
typedef int ContinuousActionsNb;
typedef int DiscreteActionsNb;

typedef int SensorsNb;
typedef int SensorsHiddenSize;

typedef int ActionsHiddenSize;

typedef int BatchSize;

class QFunctionTestParam : public testing::TestWithParam<std::tuple<
                               HiddenLayers, ContinuousActionsNb, DiscreteActionsNb, SensorsNb,
                               SensorsHiddenSize, ActionsHiddenSize, BatchSize>> {};

#endif//ARENAI_TESTS_Q_FUNCTION_H
