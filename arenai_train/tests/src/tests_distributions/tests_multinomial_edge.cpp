//
// Created by claude on 01/07/2026.
//

#include <distributions/multinomial.h>

#include <arenai_train_tests/tests_distributions/tests_multinomial_edge.h>

using namespace arenai;
using namespace arenai::train;

TEST_F(MultinomialEdgeTest, EntropyWithNearZeroProbabilities) {
    auto proba = torch::zeros({1, 5});
    proba[0][0] = 1e-10f;
    proba[0][1] = 1e-10f;
    proba[0][2] = 1e-10f;
    proba[0][3] = 1e-10f;
    proba[0][4] = 1.f - 4e-10f;

    const auto entropy = multinomial_entropy(proba);

    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>())
        << "Entropy should be finite with near-zero probabilities";
    ASSERT_GE(entropy.item<float>(), 0.0f) << "Entropy should be non-negative";
}

TEST_F(MultinomialEdgeTest, EntropyGradientFlowsThroughProbabilities) {
    auto logits = torch::randn({4, 3}, torch::TensorOptions().requires_grad(true));
    const auto proba = torch::softmax(logits, -1);

    const auto entropy = multinomial_entropy(proba);
    const auto loss = entropy.sum();

    loss.backward();

    ASSERT_TRUE(logits.grad().defined()) << "Gradient should flow back through entropy";
    ASSERT_TRUE(torch::all(torch::isfinite(logits.grad())).item<bool>())
        << "Gradient should be finite";
}

TEST_F(MultinomialEdgeTest, SampleWithSingleAction) {
    const auto proba = torch::ones({4, 1});

    const auto sample = multinomial_sample(proba);

    ASSERT_EQ(sample.size(0), 4);
    ASSERT_EQ(sample.size(1), 1);
    ASSERT_TRUE(torch::allclose(sample, torch::ones({4, 1})))
        << "Single-action sample should always be 1";
}

TEST_F(MultinomialEdgeTest, MaximumEntropyWithSingleAction) {
    const auto max_ent = multinomial_maximum_entropy(1);

    ASSERT_NEAR(max_ent, 0.0f, 1e-4f) << "Maximum entropy with 1 action should be 0 (log(1)=0)";
}

TEST_F(MultinomialEdgeTest, TargetEntropyBoundaryProbabilities) {
    const auto target_0 = multinomial_target_entropy(0.0f);
    const auto target_1 = multinomial_target_entropy(1.0f);

    ASSERT_TRUE(std::isfinite(target_0)) << "Target entropy with p=0 should be finite (clamped)";
    ASSERT_TRUE(std::isfinite(target_1)) << "Target entropy with p=1 should be finite (clamped)";
}
