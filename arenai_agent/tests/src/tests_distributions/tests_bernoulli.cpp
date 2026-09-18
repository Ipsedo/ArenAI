//
// Created by samuel on 18/09/2026.
//

#include <distributions/bernoulli.h>

#include <arenai_agent_tests/tests_distributions/tests_bernoulli.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(BernoulliTest, EntropyMaxAtHalf) {
    const auto proba = torch::full({1, 3}, 0.5f);

    const auto entropy = bernoulli_entropy(proba);

    ASSERT_TRUE(torch::allclose(entropy, torch::full({1, 3}, std::log(2.f)), 1e-4f))
        << "Each action entropy should be log(2) at p=0.5";
}

TEST_F(BernoulliTest, EntropyMinAtDegenerate) {
    const auto proba = torch::tensor({{0.f, 1.f}});

    const auto entropy = bernoulli_entropy(proba);

    ASSERT_TRUE(torch::all(torch::lt(entropy, 1e-3f)).item<bool>())
        << "Entropy should be near 0 for degenerate probabilities";
}

TEST_F(BernoulliTest, MaximumEntropyEqualsLog2) {
    ASSERT_NEAR(bernoulli_maximum_entropy(), std::log(2.f), 1e-6f);
}

TEST_F(BernoulliTest, MaxActionThreshold) {
    const auto proba = torch::tensor({{0.49f, 0.51f}, {0.9f, 0.1f}});

    const auto action = bernoulli_max_action(proba);

    ASSERT_TRUE(torch::allclose(action, torch::tensor({{0.f, 1.f}, {1.f, 0.f}})))
        << "Actions should engage strictly above the 0.5 threshold";
}

TEST_F(BernoulliTest, LogProbaMatchesTakenActions) {
    const auto proba = torch::tensor({{0.8f, 0.3f}});
    const auto action = torch::tensor({{1.f, 0.f}});

    const auto log_proba = bernoulli_log_proba(action, proba);

    ASSERT_NEAR(log_proba[0][0].item<float>(), std::log(0.8f), 1e-4f);
    ASSERT_NEAR(log_proba[0][1].item<float>(), std::log(0.7f), 1e-4f);
}

// ========================================================================
// Parameterized: sample shape and binary property
// ========================================================================

TEST_P(BernoulliShapeParamTest, SampleIsBinary) {
    const auto [batch_size, nb_actions] = GetParam();

    const auto proba = torch::sigmoid(torch::randn({batch_size, nb_actions}));

    const auto sample = bernoulli_sample(proba);

    ASSERT_EQ(sample.size(0), batch_size);
    ASSERT_EQ(sample.size(1), nb_actions);

    // each element is 0 or 1, independently of the others
    const auto is_binary = torch::logical_or(torch::eq(sample, 0.0f), torch::eq(sample, 1.0f));
    ASSERT_TRUE(torch::all(is_binary).item<bool>()) << "Sample should contain only 0s and 1s";
}

TEST_P(BernoulliShapeParamTest, EntropyShapeAndBounds) {
    const auto [batch_size, nb_actions] = GetParam();

    const auto proba = torch::sigmoid(torch::randn({batch_size, nb_actions}));

    const auto entropy = bernoulli_entropy(proba);

    ASSERT_EQ(entropy.size(0), batch_size);
    ASSERT_EQ(entropy.size(1), nb_actions);
    ASSERT_TRUE(torch::all(torch::ge(entropy, 0.0f)).item<bool>())
        << "Entropy should be non-negative";
    ASSERT_TRUE(torch::all(torch::le(entropy, bernoulli_maximum_entropy() + 1e-4f)).item<bool>())
        << "Each action entropy should be <= log(2)";
}

TEST_P(BernoulliShapeParamTest, LogProbaShapeAndFinite) {
    const auto [batch_size, nb_actions] = GetParam();

    const auto proba = torch::sigmoid(torch::randn({batch_size, nb_actions}));
    const auto action = bernoulli_sample(proba);

    const auto log_proba = bernoulli_log_proba(action, proba);

    ASSERT_EQ(log_proba.size(0), batch_size);
    ASSERT_EQ(log_proba.size(1), nb_actions);
    ASSERT_TRUE(torch::all(torch::isfinite(log_proba)).item<bool>());
    ASSERT_TRUE(torch::all(torch::le(log_proba, 0.0f)).item<bool>())
        << "Log-probabilities should be <= 0";
}

INSTANTIATE_TEST_SUITE_P(
    BernoulliShape, BernoulliShapeParamTest,
    testing::Combine(testing::Values(1, 2, 8, 16), testing::Values(1, 2, 3, 5)));
