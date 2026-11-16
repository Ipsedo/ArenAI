#include <agents/update_utils.h>
#include <gtest/gtest.h>

using namespace arenai;
using namespace arenai::agent;

TEST(ExplainedVariance, IndependentOfTheSplitIntoBlocks) {
    const auto returns = torch::randn({256, 1});
    const auto values = returns + 0.5f * torch::randn({256, 1});

    const float whole = explained_variance({returns}, {values});
    const float split = explained_variance(returns.split(32), values.split(32));

    ASSERT_NEAR(whole, split, 1e-5f);
    ASSERT_GT(whole, 0.5f);
}

TEST(ExplainedVariance, PerfectAndMeanPredictions) {
    const auto returns = torch::randn({64, 1});

    ASSERT_NEAR(explained_variance({returns}, {returns.clone()}), 1.f, 1e-6f);
    ASSERT_NEAR(
        explained_variance({returns}, {torch::full_like(returns, returns.mean().item<float>())}),
        0.f, 1e-5f);
}
