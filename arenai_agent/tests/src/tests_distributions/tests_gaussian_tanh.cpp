#include <distributions/gaussian_tanh.h>

#include <arenai_agent_tests/tests_distributions/tests_gaussian_tanh.h>

using namespace arenai;
using namespace arenai::agent;

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

TEST_P(GaussianTanhShapeParamTest, SampleShapeAndBounds) {
    const auto &shape = GetParam();

    const auto mu = torch::randn(shape);
    const auto sigma = torch::rand(shape) + 0.1f;

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);

    ASSERT_EQ(action.sizes(), mu.sizes());
    ASSERT_EQ(u.sizes(), mu.sizes());
    ASSERT_TRUE(torch::all(torch::logical_and(torch::gt(action, -1.0f), torch::lt(action, 1.0f)))
                    .item<bool>());
}

TEST_P(GaussianTanhShapeParamTest, LogPdfShapeAndFinite) {
    const auto &shape = GetParam();

    const auto mu = torch::randn(shape);
    const auto sigma = torch::rand(shape) + 0.1f;

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);
    const auto log_p = gaussian_tanh_log_pdf(u, mu, sigma);

    ASSERT_EQ(log_p.sizes(), mu.sizes());
    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>());
}

TEST_P(GaussianTanhShapeParamTest, SampleFiniteWithZeroMu) {
    const auto &shape = GetParam();

    const auto mu = torch::zeros(shape);
    const auto sigma = torch::ones(shape) * 0.5f;

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(action)).item<bool>());
    ASSERT_TRUE(torch::all(torch::isfinite(u)).item<bool>());
}

TEST_P(GaussianTanhShapeParamTest, SampleFiniteWithLargeMu) {
    const auto &shape = GetParam();

    const auto mu = torch::ones(shape) * 10.0f;
    const auto sigma = torch::ones(shape) * 0.1f;

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(action)).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    GaussianTanh, GaussianTanhShapeParamTest,
    testing::Values(Shape{1}, Shape{5}, Shape{2, 3}, Shape{4, 8}, Shape{16, 4}));

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

TEST_F(GaussianTanhTest, ModeIsTanhOfMu) {
    const auto mu = torch::randn({100}) * 3.f;

    ASSERT_TRUE(torch::allclose(gaussian_tanh_mode(mu), torch::tanh(mu)));
}

TEST_P(GaussianTanhShapeParamTest, EntropyShapeAndFinite) {
    const auto &shape = GetParam();

    const auto mu = torch::randn(shape);
    const auto sigma = torch::rand(shape) + 0.1f;

    const auto entropy = gaussian_tanh_entropy(mu, sigma);

    ASSERT_EQ(entropy.sizes(), mu.sizes());
    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>());
}

TEST_P(GaussianTanhTargetEntropyParamTest, EntropyEstimateMatchesTargetEntropy) {
    const auto [nb_actions, target_sigma] = GetParam();

    constexpr int nb_rows = 200000;
    const auto mu = torch::zeros({nb_rows, nb_actions});
    const auto sigma = torch::full({nb_rows, nb_actions}, target_sigma);

    const auto estimate = gaussian_tanh_entropy(mu, sigma).sum(-1).mean().item<float>();
    const auto target = gaussian_tanh_target_entropy(nb_actions, target_sigma);

    ASSERT_NEAR(estimate, target, 0.02f * static_cast<float>(nb_actions));
}

TEST_F(GaussianTanhTest, EntropyGrowsWithSigma) {
    constexpr int nb_rows = 100000;
    const auto mu = torch::zeros({nb_rows});

    const auto small = gaussian_tanh_entropy(mu, torch::full({nb_rows}, 0.1f)).mean().item<float>();
    const auto large = gaussian_tanh_entropy(mu, torch::full({nb_rows}, 0.5f)).mean().item<float>();

    ASSERT_LT(small, large);
}

TEST_F(GaussianTanhTest, EntropyDropsNearTheBounds) {
    constexpr int nb_rows = 100000;
    const auto sigma = torch::full({nb_rows}, 0.3f);

    const auto centered =
        gaussian_tanh_entropy(torch::zeros({nb_rows}), sigma).mean().item<float>();
    const auto saturated =
        gaussian_tanh_entropy(torch::full({nb_rows}, 3.f), sigma).mean().item<float>();

    ASSERT_LT(saturated, centered);
}

TEST_F(GaussianTanhTest, EntropyGradientFlowsThroughMuAndSigma) {
    const auto mu = torch::full({64}, 0.5f, torch::TensorOptions().requires_grad(true));
    const auto sigma = torch::full({64}, 0.3f, torch::TensorOptions().requires_grad(true));

    gaussian_tanh_entropy(mu, sigma).mean().backward();

    ASSERT_TRUE(torch::all(torch::isfinite(mu.grad())).item<bool>());
    ASSERT_TRUE(torch::all(torch::isfinite(sigma.grad())).item<bool>());
    // widening sigma raises the entropy: the gradient of the mean entropy is positive
    ASSERT_GT(sigma.grad().sum().item<float>(), 0.f);
}

TEST_F(GaussianTanhTest, KlIsZeroBetweenIdenticalDistributions) {
    const auto mu = torch::randn({100});
    const auto sigma = torch::rand({100}) + 0.1f;

    const auto kl = gaussian_tanh_kl(mu, sigma, mu, sigma);

    ASSERT_TRUE(torch::allclose(kl, torch::zeros_like(kl), 1e-5, 1e-6));
}

TEST_F(GaussianTanhTest, KlIsPositiveBetweenDifferentDistributions) {
    const auto mu_p = torch::randn({100});
    const auto sigma_p = torch::rand({100}) + 0.1f;
    const auto mu_q = mu_p + torch::randn({100});
    const auto sigma_q = torch::rand({100}) + 0.1f;

    const auto kl = gaussian_tanh_kl(mu_p, sigma_p, mu_q, sigma_q);

    ASSERT_TRUE(torch::all(torch::gt(kl, 0.f)).item<bool>());
}

TEST_F(GaussianTanhTest, KlMatchesClosedForm) {
    // KL(N(0, 1) || N(1, 2)) = log 2 + (1 + 1) / 8 - 1/2
    const auto kl = gaussian_tanh_kl(
        torch::zeros({1}), torch::ones({1}), torch::ones({1}), torch::full({1}, 2.f));

    ASSERT_NEAR(kl.item<float>(), std::log(2.f) + 0.25f - 0.5f, 1e-5f);
}
