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
    // concentration=2 → alpha=beta=1 → uniform → maximal entropy
    const auto entropy_uniform = beta_law_entropy(torch::tensor({0.5f}), torch::tensor({2.0f}));
    const auto entropy_peaked = beta_law_entropy(torch::tensor({0.5f}), torch::tensor({10.0f}));

    ASSERT_GT(entropy_uniform.item<float>(), entropy_peaked.item<float>());
}

TEST_F(BetaLawTest, EntropyDecreasesWithConcentration) {
    const auto mode = torch::tensor({0.3f});

    const auto entropy_low = beta_law_entropy(mode, torch::tensor({5.0f}));
    const auto entropy_high = beta_law_entropy(mode, torch::tensor({50.0f}));

    ASSERT_GT(entropy_low.item<float>(), entropy_high.item<float>());
}

TEST_F(BetaLawTest, TargetEntropyProportionalToActions) {
    const auto t1 = beta_law_target_entropy(1);
    const auto t3 = beta_law_target_entropy(3);

    ASSERT_NEAR(t3, 3.0f * t1, 1e-5f);
}

TEST_F(BetaLawTest, LogProbaConsistentWithSample) {
    const auto mode = torch::ones({100}) * 0.4f;
    const auto concentration = torch::ones({100}) * 5.0f;

    const auto samples = beta_law_sample(mode, concentration);
    const auto log_p = beta_law_log_proba(samples, mode, concentration);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>());
}

TEST_F(BetaLawTest, MeanActionMatchesModeWhenCentered) {
    // mode=0.5 → alpha=beta → mean action 0 on [-1, 1]
    const auto mode = torch::ones({10}) * 0.5f;
    const auto concentration = torch::ones({10}) * 8.0f;

    const auto mean = beta_law_mean_action(mode, concentration);

    ASSERT_TRUE(torch::allclose(mean, torch::zeros({10}), 1e-5, 1e-5));
}

// ========================================================================
// Parameterized: shape variations
// ========================================================================

TEST_P(BetaLawParamTest, SampleBounds) {
    const auto &shape = GetParam();

    const auto mode = torch::rand(shape);
    const auto concentration = torch::rand(shape) * 8.0f + 2.0f;

    const auto samples = beta_law_sample(mode, concentration);

    ASSERT_EQ(samples.sizes(), mode.sizes());
    ASSERT_TRUE(torch::all(torch::logical_and(torch::ge(samples, -1.0f), torch::le(samples, 1.0f)))
                    .item<bool>());
}

TEST_P(BetaLawParamTest, LogProbaShape) {
    const auto &shape = GetParam();

    const auto mode = torch::rand(shape);
    const auto concentration = torch::rand(shape) * 8.0f + 2.0f;
    const auto samples = beta_law_sample(mode, concentration);

    const auto log_p = beta_law_log_proba(samples, mode, concentration);

    ASSERT_EQ(log_p.sizes(), samples.sizes());
    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>());
}

TEST_P(BetaLawParamTest, EntropyShape) {
    const auto &shape = GetParam();

    const auto mode = torch::rand(shape);
    const auto concentration = torch::rand(shape) * 8.0f + 2.0f;

    const auto entropy = beta_law_entropy(mode, concentration);

    ASSERT_EQ(entropy.sizes(), mode.sizes());
    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>());
}

TEST_P(BetaLawParamTest, SampleNoNaNBelowUniformConcentration) {
    const auto &shape = GetParam();

    // concentration below the κ=2 floor → alpha/beta below the clamp, still finite
    const auto mode = torch::ones(shape) * 0.5f;
    const auto concentration = torch::ones(shape) * 0.1f;

    const auto samples = beta_law_sample(mode, concentration);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>());
    ASSERT_TRUE(torch::all(torch::logical_and(torch::ge(samples, -1.0f), torch::le(samples, 1.0f)))
                    .item<bool>());
}

TEST_P(BetaLawParamTest, SampleNoNaNWithLargeConcentration) {
    const auto &shape = GetParam();

    const auto mode = torch::ones(shape) * 0.5f;
    const auto concentration = torch::ones(shape) * 2000.0f;

    const auto samples = beta_law_sample(mode, concentration);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    BetaLaw, BetaLawParamTest,
    testing::Values(Shape{1}, Shape{5}, Shape{2, 3}, Shape{4, 8}, Shape{16, 4}));
