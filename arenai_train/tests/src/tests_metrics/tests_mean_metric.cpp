//
// Created by samuel on 30/06/2026.
//

#include <metrics/mean_metric.h>

#include <arenai_train_tests/tests_metrics/tests_metrics.h>

using namespace arenai;
using namespace arenai::train;

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(MeanMetricTest, KnownSequence) {
    MeanMetric metric("test", 5);

    for (const float v: {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}) metric.add(v);

    ASSERT_NEAR(metric.compute_metric(), 3.0f, 1e-6f);
}

TEST_F(MeanMetricTest, SingleValue) {
    MeanMetric metric("test", 10);

    metric.add(42.0f);

    ASSERT_NEAR(metric.compute_metric(), 42.0f, 1e-6f);
}

TEST_F(MeanMetricTest, NegativeValues) {
    MeanMetric metric("test", 3);

    metric.add(-3.0f);
    metric.add(-6.0f);
    metric.add(-9.0f);

    ASSERT_NEAR(metric.compute_metric(), -6.0f, 1e-6f);
}

// ========================================================================
// Parameterized: window eviction behavior
// ========================================================================

TEST_P(MeanMetricParamTest, ConstantValueAlwaysReturnsSame) {
    const auto [window_size, values_to_add] = GetParam();

    MeanMetric metric("test", window_size);

    constexpr float constant = 7.0f;
    for (int i = 0; i < values_to_add; i++) metric.add(constant);

    ASSERT_NEAR(metric.compute_metric(), constant, 1e-6f);
}

TEST_P(MeanMetricParamTest, WindowEviction) {
    const auto [window_size, values_to_add] = GetParam();

    MeanMetric metric("test", window_size);

    // fill with 100s then overwrite with 1s
    for (int i = 0; i < window_size; i++) metric.add(100.0f);
    for (int i = 0; i < values_to_add; i++) metric.add(1.0f);

    if (values_to_add >= window_size) {
        ASSERT_NEAR(metric.compute_metric(), 1.0f, 1e-6f);
    } else {
        const float expected = (100.0f * static_cast<float>(window_size - values_to_add)
                                + 1.0f * static_cast<float>(values_to_add))
                               / static_cast<float>(window_size);
        ASSERT_NEAR(metric.compute_metric(), expected, 1e-4f);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MeanMetric, MeanMetricParamTest,
    testing::Combine(testing::Values(3, 5, 10, 50), testing::Values(1, 3, 5, 10, 50, 100)));
