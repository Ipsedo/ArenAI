//
// Created by samuel on 30/06/2026.
//

#include <distributions/multinomial.h>

#include <arenai_train_tests/tests_distributions/tests_multinomial.h>

using namespace arenai;
using namespace arenai::train;

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(MultinomialTest, EntropyMaxAtUniform) {
    constexpr int n = 5;
    const auto uniform = torch::ones({1, n}) / static_cast<float>(n);

    const auto entropy = multinomial_entropy(uniform);

    ASSERT_NEAR(entropy.item<float>(), std::log(static_cast<float>(n)), 1e-4f);
}

TEST_F(MultinomialTest, EntropyMinAtDegenerate) {
    const auto proba = torch::zeros({1, 4});
    proba[0][0] = 1.0f;

    const auto entropy = multinomial_entropy(proba);

    ASSERT_NEAR(entropy.item<float>(), 0.0f, 1e-3f);
}

TEST_F(MultinomialTest, TargetEntropySymmetric) {
    const auto target = multinomial_target_entropy(0.5f);
    const auto max_2 = multinomial_maximum_entropy(2);

    ASSERT_NEAR(target, max_2, 1e-5f);
}

TEST_F(MultinomialTest, MaxEntropyEqualsLogN) {
    for (const int n: {2, 3, 5, 10}) {
        const auto max_ent = multinomial_maximum_entropy(n);
        ASSERT_NEAR(max_ent, std::log(static_cast<float>(n)), 1e-4f)
            << "Maximum entropy for n=" << n << " should be log(n)";
    }
}

// ========================================================================
// Parameterized: sample shape and one-hot property
// ========================================================================

TEST_P(MultinomialShapeParamTest, SampleIsOneHot) {
    const auto [batch_size, nb_actions] = GetParam();

    const auto proba = torch::softmax(torch::randn({batch_size, nb_actions}), -1);

    const auto sample = multinomial_sample(proba);

    ASSERT_EQ(sample.size(0), batch_size);
    ASSERT_EQ(sample.size(1), nb_actions);

    // each row sums to 1
    const auto row_sums = torch::sum(sample, -1);
    ASSERT_TRUE(torch::allclose(row_sums, torch::ones({batch_size})))
        << "Each sample row should sum to 1";

    // each element is 0 or 1
    const auto is_binary = torch::logical_or(torch::eq(sample, 0.0f), torch::eq(sample, 1.0f));
    ASSERT_TRUE(torch::all(is_binary).item<bool>()) << "Sample should contain only 0s and 1s";
}

TEST_P(MultinomialShapeParamTest, EntropyShape) {
    const auto [batch_size, nb_actions] = GetParam();

    const auto proba = torch::softmax(torch::randn({batch_size, nb_actions}), -1);

    const auto entropy = multinomial_entropy(proba);

    ASSERT_EQ(entropy.size(0), batch_size);
    ASSERT_EQ(entropy.size(1), 1);
    ASSERT_TRUE(torch::all(torch::ge(entropy, 0.0f)).item<bool>())
        << "Entropy should be non-negative";
    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>());
}

TEST_P(MultinomialShapeParamTest, EntropyBoundedByLogN) {
    const auto [batch_size, nb_actions] = GetParam();

    const auto proba = torch::softmax(torch::randn({batch_size, nb_actions}), -1);

    const auto entropy = multinomial_entropy(proba);
    const auto max_entropy = std::log(static_cast<float>(nb_actions));

    ASSERT_TRUE(torch::all(torch::le(entropy, max_entropy + 1e-4f)).item<bool>())
        << "Entropy should be <= log(nb_actions)";
}

INSTANTIATE_TEST_SUITE_P(
    MultinomialShape, MultinomialShapeParamTest,
    testing::Combine(testing::Values(1, 2, 8, 16), testing::Values(2, 3, 4, 5, 10)));

// ========================================================================
// Parameterized: maximum entropy monotone
// ========================================================================

TEST_P(MultinomialMaxEntropyParamTest, MonotoneIncreasing) {
    const auto nb_actions = GetParam();

    if (nb_actions < 2) return;

    const auto ent_n = multinomial_maximum_entropy(nb_actions);
    const auto ent_n_minus_1 = multinomial_maximum_entropy(nb_actions - 1);

    ASSERT_GT(ent_n, ent_n_minus_1);
}

INSTANTIATE_TEST_SUITE_P(
    MultinomialMaxEntropy, MultinomialMaxEntropyParamTest, testing::Values(2, 3, 4, 5, 10, 20));

// ========================================================================
// Parameterized: target entropy
// ========================================================================

TEST_P(MultinomialTargetEntropyParamTest, TargetEntropyBounded) {
    const auto shoot_probability = GetParam();

    const auto target = multinomial_target_entropy(shoot_probability);

    ASSERT_GE(target, 0.0f);
    ASSERT_LE(target, multinomial_maximum_entropy(2) + 1e-5f);
    ASSERT_TRUE(std::isfinite(target));
}

TEST_P(MultinomialTargetEntropyParamTest, HigherProbabilityLowerEntropy) {
    const auto shoot_probability = GetParam();

    if (shoot_probability >= 0.5f) return;

    const auto target_low = multinomial_target_entropy(shoot_probability);
    const auto target_half = multinomial_target_entropy(0.5f);

    ASSERT_LT(target_low, target_half);
}

INSTANTIATE_TEST_SUITE_P(
    MultinomialTargetEntropy, MultinomialTargetEntropyParamTest,
    testing::Values(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.8f, 0.9f));
