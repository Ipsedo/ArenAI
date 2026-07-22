//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_METRICS_H
#define ARENAI_TESTS_METRICS_H

#include <gtest/gtest.h>

typedef int WindowSize;
typedef int ValuesToAdd;

class MeanMetricTest : public testing::Test {};
class MeanMetricParamTest : public testing::TestWithParam<std::tuple<WindowSize, ValuesToAdd>> {};

class StdMetricTest : public testing::Test {};
class StdMetricParamTest : public testing::TestWithParam<std::tuple<WindowSize, ValuesToAdd>> {};

class AbstractMetricTest : public testing::Test {};
class AbstractMetricWindowParamTest : public testing::TestWithParam<WindowSize> {};

class MetricCsvSaverTest : public testing::Test {};
class MetricCsvSaverParamTest
    : public testing::TestWithParam<std::tuple<WindowSize, int /*save_every*/, int /*calls*/>> {};

#endif//ARENAI_TESTS_METRICS_H
