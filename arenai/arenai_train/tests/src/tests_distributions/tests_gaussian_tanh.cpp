//
// Created by samuel on 30/06/2026.
//

#include <distributions/gaussian_tanh.h>

#include <arenai_train_tests/tests_distributions/tests_gaussian_tanh.h>

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(GaussianTanhTest, ActionBoundedByTanh) {
    const auto mu = torch::zeros({1000});
    const auto sigma = torch::ones({1000});

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);

    ASSERT_TRUE(torch::all(torch::logical_and(torch::gt(action, -1.0f), torch::lt(action, 1.0f)))
                    .item<bool>());
}

TEST_F(GaussianTanhTest, LogPdfConsistentWithSample) {
    const auto mu = torch::randn({100});
    const auto sigma = torch::rand({100}) + 0.1f;

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);
    const auto log_p = gaussian_tanh_log_pdf(u, mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>());
}

TEST_F(GaussianTanhTest, SmallSigmaConcentratesAction) {
    const auto mu = torch::zeros({1000});
    const auto small_sigma = torch::ones({1000}) * 0.01f;
    const auto large_sigma = torch::ones({1000}) * 1.0f;

    const auto [action_small, _u1] = gaussian_tanh_sample(mu, small_sigma);
    const auto [action_large, _u2] = gaussian_tanh_sample(mu, large_sigma);

    const auto std_small = torch::std(action_small).item<float>();
    const auto std_large = torch::std(action_large).item<float>();

    ASSERT_LT(std_small, std_large);
}

// ========================================================================
// Parameterized: shape variations
// ========================================================================

TEST_P(GaussianTanhShapeParamTest, SampleShapeAndBounds) {
    const auto shape = GetParam();

    const auto mu = torch::randn(shape);
    const auto sigma = torch::rand(shape) + 0.1f;

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);

    ASSERT_EQ(action.sizes(), mu.sizes());
    ASSERT_EQ(u.sizes(), mu.sizes());
    ASSERT_TRUE(torch::all(torch::logical_and(torch::gt(action, -1.0f), torch::lt(action, 1.0f)))
                    .item<bool>());
}

TEST_P(GaussianTanhShapeParamTest, LogPdfShapeAndFinite) {
    const auto shape = GetParam();

    const auto mu = torch::randn(shape);
    const auto sigma = torch::rand(shape) + 0.1f;

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);
    const auto log_p = gaussian_tanh_log_pdf(u, mu, sigma);

    ASSERT_EQ(log_p.sizes(), mu.sizes());
    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>());
}

TEST_P(GaussianTanhShapeParamTest, SampleFiniteWithZeroMu) {
    const auto shape = GetParam();

    const auto mu = torch::zeros(shape);
    const auto sigma = torch::ones(shape) * 0.5f;

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(action)).item<bool>());
    ASSERT_TRUE(torch::all(torch::isfinite(u)).item<bool>());
}

TEST_P(GaussianTanhShapeParamTest, SampleFiniteWithLargeMu) {
    const auto shape = GetParam();

    const auto mu = torch::ones(shape) * 10.0f;
    const auto sigma = torch::ones(shape) * 0.1f;

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(action)).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    GaussianTanh, GaussianTanhShapeParamTest,
    testing::Values(Shape{1}, Shape{5}, Shape{2, 3}, Shape{4, 8}, Shape{16, 4}));

// ========================================================================
// Parameterized: target entropy
// ========================================================================

TEST_P(GaussianTanhTargetEntropyParamTest, TargetEntropyFinite) {
    const auto [nb_actions, target_sigma] = GetParam();

    const auto entropy = gaussian_tanh_target_entropy(nb_actions, target_sigma);

    ASSERT_TRUE(std::isfinite(entropy));
}

TEST_P(GaussianTanhTargetEntropyParamTest, TargetEntropyScalesWithActions) {
    const auto [nb_actions, target_sigma] = GetParam();

    if (nb_actions < 2) return;

    const auto entropy_n = gaussian_tanh_target_entropy(nb_actions, target_sigma);
    const auto entropy_1 = gaussian_tanh_target_entropy(1, target_sigma);

    ASSERT_NEAR(entropy_n, static_cast<float>(nb_actions) * entropy_1, std::abs(entropy_1) * 0.1f);
}

INSTANTIATE_TEST_SUITE_P(
    GaussianTanhTarget, GaussianTanhTargetEntropyParamTest,
    testing::Combine(testing::Values(1, 2, 3, 5), testing::Values(0.1f, 0.5f, 1.0f)));
