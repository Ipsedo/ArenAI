//
// Created by claude on 01/07/2026.
//

#include <metrics/mean_metric.h>
#include <metrics/std_metric.h>

#include <arenai_train_tests/tests_metrics/tests_metrics_edge.h>

TEST_F(MetricsEdgeTest, MeanMetricComputeOnEmpty) {
    MeanMetric metric("test_mean", 10);

    const auto result = metric.compute_metric();

    ASSERT_TRUE(std::isfinite(result))
        << "compute_metric on empty MeanMetric should not produce NaN/Inf";
}

TEST_F(MetricsEdgeTest, MeanMetricLastValueOnEmpty) {
    const MeanMetric metric("test_mean", 10);

    ASSERT_DEATH(metric.last_value(), "")
        << "last_value on empty metric triggers UB (vector.back() on empty)";
}

TEST_F(MetricsEdgeTest, StdMetricComputeOnEmpty) {
    StdMetric metric("test_std", 10);

    const auto result = metric.compute_metric();

    ASSERT_TRUE(std::isfinite(result))
        << "compute_metric on empty StdMetric should not produce NaN/Inf";
}

TEST_F(MetricsEdgeTest, StdMetricLastValueOnEmpty) {
    const StdMetric metric("test_std", 10);

    ASSERT_DEATH(metric.last_value(), "")
        << "last_value on empty StdMetric triggers UB (vector.back() on empty)";
}

TEST_F(MetricsEdgeTest, MeanMetricSingleValue) {
    MeanMetric metric("test_mean", 10);

    metric.add(42.0f);

    ASSERT_NEAR(metric.compute_metric(), 42.0f, 1e-5f);
    ASSERT_NEAR(metric.last_value(), 42.0f, 1e-5f);
}

TEST_F(MetricsEdgeTest, StdMetricSingleValue) {
    StdMetric metric("test_std", 10);

    metric.add(42.0f);

    const auto result = metric.compute_metric();
    ASSERT_TRUE(std::isfinite(result));
    ASSERT_NEAR(result, 0.0f, 1e-5f) << "Std of single value should be 0";
}

TEST_F(MetricsEdgeTest, MeanMetricWindowSliding) {
    MeanMetric metric("test_mean", 3);

    metric.add(1.0f);
    metric.add(2.0f);
    metric.add(3.0f);
    ASSERT_NEAR(metric.compute_metric(), 2.0f, 1e-5f);

    metric.add(10.0f);
    ASSERT_NEAR(metric.compute_metric(), 5.0f, 1e-5f)
        << "Window should slide: mean of {2, 3, 10} = 5";
}

TEST_F(MetricsEdgeTest, MeanMetricWithConstantValues) {
    MeanMetric metric("test_mean", 5);

    for (int i = 0; i < 20; i++) metric.add(7.0f);

    ASSERT_NEAR(metric.compute_metric(), 7.0f, 1e-5f);
}

TEST_F(MetricsEdgeTest, StdMetricWithConstantValues) {
    StdMetric metric("test_std", 5);

    for (int i = 0; i < 20; i++) metric.add(3.0f);

    ASSERT_NEAR(metric.compute_metric(), 0.0f, 1e-5f) << "Std of constant values should be 0";
}

TEST_F(MetricsEdgeTest, ToStringOnEmptyDoesNotCrash) {
    MeanMetric metric("test_mean", 10);

    std::string s;
    ASSERT_NO_THROW(s = metric.to_string());
    ASSERT_FALSE(s.empty());
}
