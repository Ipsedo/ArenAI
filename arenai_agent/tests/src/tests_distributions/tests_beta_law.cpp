//
// Created by samuel on 30/06/2026.
//

#include <distributions/beta_law.h>

#include <arenai_agent_tests/tests_distributions/tests_beta_law.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(BetaLawTest, UniformEntropyIsMaximal) {
    // alpha=1, beta=1 → Kumaraswamy uniform → maximal entropy
    const auto entropy_uniform = beta_law_entropy(torch::tensor({1.0f}), torch::tensor({1.0f}));
    const auto entropy_peaked = beta_law_entropy(torch::tensor({5.0f}), torch::tensor({5.0f}));

    ASSERT_GT(entropy_uniform.item<float>(), entropy_peaked.item<float>());
}

TEST_F(BetaLawTest, TargetEntropyProportionalToActions) {
    const auto t1 = beta_law_target_entropy(1);
    const auto t3 = beta_law_target_entropy(3);

    ASSERT_NEAR(t3, 3.0f * t1, 1e-5f);
}

TEST_F(BetaLawTest, LogProbaConsistentWithSample) {
    const auto alpha = torch::ones({100}) * 2.0f;
    const auto beta = torch::ones({100}) * 3.0f;

    const auto samples = beta_law_sample(alpha, beta);
    const auto log_p = beta_law_log_proba(samples, alpha, beta);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>());
}

// ========================================================================
// Parameterized: shape variations
// ========================================================================

TEST_P(BetaLawParamTest, SampleBounds) {
    const auto shape = GetParam();

    const auto alpha = torch::rand(shape) * 4.0f + 0.5f;
    const auto beta = torch::rand(shape) * 4.0f + 0.5f;

    const auto samples = beta_law_sample(alpha, beta);

    ASSERT_EQ(samples.sizes(), alpha.sizes());
    ASSERT_TRUE(torch::all(torch::logical_and(torch::ge(samples, -1.0f), torch::le(samples, 1.0f)))
                    .item<bool>());
}

TEST_P(BetaLawParamTest, LogProbaShape) {
    const auto shape = GetParam();

    const auto alpha = torch::rand(shape) * 4.0f + 0.5f;
    const auto beta = torch::rand(shape) * 4.0f + 0.5f;
    const auto samples = beta_law_sample(alpha, beta);

    const auto log_p = beta_law_log_proba(samples, alpha, beta);

    ASSERT_EQ(log_p.sizes(), samples.sizes());
    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>());
}

TEST_P(BetaLawParamTest, EntropyShape) {
    const auto shape = GetParam();

    const auto alpha = torch::rand(shape) * 4.0f + 0.5f;
    const auto beta = torch::rand(shape) * 4.0f + 0.5f;

    const auto entropy = beta_law_entropy(alpha, beta);

    ASSERT_EQ(entropy.sizes(), alpha.sizes());
    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>());
}

TEST_P(BetaLawParamTest, SampleNoNaNWithSmallParams) {
    const auto shape = GetParam();

    const auto alpha = torch::ones(shape) * 0.1f;
    const auto beta = torch::ones(shape) * 0.1f;

    const auto samples = beta_law_sample(alpha, beta);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>());
    ASSERT_TRUE(torch::all(torch::logical_and(torch::ge(samples, -1.0f), torch::le(samples, 1.0f)))
                    .item<bool>());
}

TEST_P(BetaLawParamTest, SampleNoNaNWithLargeParams) {
    const auto shape = GetParam();

    const auto alpha = torch::ones(shape) * 50.0f;
    const auto beta = torch::ones(shape) * 50.0f;

    const auto samples = beta_law_sample(alpha, beta);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    BetaLaw, BetaLawParamTest,
    testing::Values(Shape{1}, Shape{5}, Shape{2, 3}, Shape{4, 8}, Shape{16, 4}));
