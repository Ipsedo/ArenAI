//
// Created by samuel on 30/06/2026.
//

#include <metrics/mean_metric.h>

#include <arenai_train_tests/tests_metrics/tests_metrics.h>

// ========================================================================
// Fixed tests (using MeanMetric as concrete impl)
// ========================================================================

TEST_F(AbstractMetricTest, LastValueReturnsLastAdded) {
    MeanMetric metric("test", 10);

    metric.add(1.0f);
    metric.add(2.0f);
    metric.add(99.0f);

    ASSERT_NEAR(metric.last_value(), 99.0f, 1e-6f);
}

TEST_F(AbstractMetricTest, GetNameReturnsName) {
    const MeanMetric metric("my_metric", 5);

    ASSERT_EQ(metric.get_name(), "my_metric");
}

TEST_F(AbstractMetricTest, ToStringFormat) {
    MeanMetric metric("loss", 3, 2, false);

    metric.add(1.0f);
    metric.add(2.0f);
    metric.add(3.0f);

    const auto str = metric.to_string();

    ASSERT_TRUE(str.find("loss") != std::string::npos);
    ASSERT_TRUE(str.find("=") != std::string::npos);
    ASSERT_TRUE(str.find("2.00") != std::string::npos);
}

TEST_F(AbstractMetricTest, ToStringScientific) {
    MeanMetric metric("tiny", 1, 2, true);

    metric.add(0.001f);

    const auto str = metric.to_string();

    ASSERT_TRUE(str.find("e") != std::string::npos || str.find("E") != std::string::npos);
}

TEST_F(AbstractMetricTest, MetricsToStringMultiple) {
    auto m1 = std::make_shared<MeanMetric>("a", 5);
    auto m2 = std::make_shared<MeanMetric>("b", 5);

    m1->add(1.0f);
    m2->add(2.0f);

    const auto str = AbstractMetric::metrics_to_string({m1, m2});

    ASSERT_TRUE(str.find("a") != std::string::npos);
    ASSERT_TRUE(str.find("b") != std::string::npos);
    ASSERT_TRUE(str.find(",") != std::string::npos);
}

// ========================================================================
// Parameterized: window size behavior
// ========================================================================

TEST_P(AbstractMetricWindowParamTest, WindowSizeRespected) {
    const auto window_size = GetParam();

    MeanMetric metric("test", window_size);

    // add 2x window_size values: first half 0, second half 10
    for (int i = 0; i < window_size; i++) metric.add(0.0f);
    for (int i = 0; i < window_size; i++) metric.add(10.0f);

    // window should contain only 10s now
    ASSERT_NEAR(metric.compute_metric(), 10.0f, 1e-6f);
}

TEST_P(AbstractMetricWindowParamTest, LastValueAfterManyAdds) {
    const auto window_size = GetParam();

    MeanMetric metric("test", window_size);

    for (int i = 0; i < window_size * 3; i++) metric.add(static_cast<float>(i));

    ASSERT_NEAR(metric.last_value(), static_cast<float>(window_size * 3 - 1), 1e-6f);
}

INSTANTIATE_TEST_SUITE_P(
    AbstractMetric, AbstractMetricWindowParamTest, testing::Values(1, 3, 5, 10, 50, 100));
