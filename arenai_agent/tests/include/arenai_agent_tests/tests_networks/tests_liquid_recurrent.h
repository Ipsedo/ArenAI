//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_TESTS_LIQUID_RECURRENT_H
#define ARENAI_TESTS_LIQUID_RECURRENT_H

#include <gtest/gtest.h>

typedef int NeuronNumber;
typedef int InputSize;
typedef int OutputSize;
typedef int UnfoldingSteps;
typedef int BatchSize;
typedef int TimeSteps;

class CellModelTestParam
    : public testing::TestWithParam<std::tuple<NeuronNumber, InputSize, BatchSize>> {};

class LiquidCellTestParam : public testing::TestWithParam<
                                std::tuple<NeuronNumber, InputSize, UnfoldingSteps, BatchSize>> {};

class LiquidRecurrentTestParam
    : public testing::TestWithParam<
          std::tuple<NeuronNumber, InputSize, OutputSize, UnfoldingSteps, BatchSize, TimeSteps>> {};

#endif//ARENAI_TESTS_LIQUID_RECURRENT_H
