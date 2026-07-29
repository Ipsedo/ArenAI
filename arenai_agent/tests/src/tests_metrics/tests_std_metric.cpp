//
// Created by samuel on 30/06/2026.
//

#include <metrics/std_metric.h>

#include <arenai_agent_tests/tests_metrics/tests_metrics.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(StdMetricTest, ConstantValuesZeroStd) {
    StdMetric metric("test", 10);

    for (int i = 0; i < 10; i++) metric.add(5.0f);

    ASSERT_NEAR(metric.compute_metric(), 0.0f, 1e-6f);
}

TEST_F(StdMetricTest, KnownSequence) {
    StdMetric metric("test", 4);

    // values: 2, 4, 4, 4 → mean = 3.5, var = (2.25+0.25+0.25+0.25)/4 = 0.75
    metric.add(2.0f);
    metric.add(4.0f);
    metric.add(4.0f);
    metric.add(4.0f);

    const float expected_std = std::sqrt(0.75f);
    ASSERT_NEAR(metric.compute_metric(), expected_std, 1e-4f);
}

TEST_F(StdMetricTest, TwoValues) {
    StdMetric metric("test", 2);

    metric.add(0.0f);
    metric.add(10.0f);

    // mean = 5, var = (25 + 25) / 2 = 25, std = 5
    ASSERT_NEAR(metric.compute_metric(), 5.0f, 1e-4f);
}

// ========================================================================
// Parameterized: constant values always give std = 0
// ========================================================================

TEST_P(StdMetricParamTest, ConstantAlwaysZero) {
    const auto [window_size, values_to_add] = GetParam();

    StdMetric metric("test", window_size);

    for (int i = 0; i < values_to_add; i++) metric.add(42.0f);

    ASSERT_NEAR(metric.compute_metric(), 0.0f, 1e-5f);
}

TEST_P(StdMetricParamTest, WindowEvictionConverges) {
    const auto [window_size, values_to_add] = GetParam();

    StdMetric metric("test", window_size);

    // fill with mixed values then overwrite entirely with constant
    for (int i = 0; i < window_size; i++) metric.add(static_cast<float>(i * 10));
    for (int i = 0; i < values_to_add; i++) metric.add(3.0f);

    if (values_to_add >= window_size) {
        ASSERT_NEAR(metric.compute_metric(), 0.0f, 1e-5f);
    } else {
        ASSERT_GE(metric.compute_metric(), 0.0f);
    }
}

INSTANTIATE_TEST_SUITE_P(
    StdMetric, StdMetricParamTest,
    testing::Combine(testing::Values(3, 5, 10, 50), testing::Values(1, 3, 5, 10, 50, 100)));
