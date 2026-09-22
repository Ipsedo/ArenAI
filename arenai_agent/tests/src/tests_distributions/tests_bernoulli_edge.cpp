//
// Created by samuel on 18/09/2026.
//

#include <distributions/bernoulli.h>

#include <arenai_agent_tests/tests_distributions/tests_bernoulli_edge.h>

using namespace arenai;
using namespace arenai::agent;

TEST_F(BernoulliEdgeTest, EntropyWithBoundaryProbabilities) {
    const auto proba = torch::tensor({{0.f, 1.f, 1e-10f, 1.f - 1e-10f}});

    const auto entropy = bernoulli_entropy(proba);

    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>())
        << "Entropy should be finite with boundary probabilities";
    ASSERT_TRUE(torch::all(torch::ge(entropy, 0.0f)).item<bool>())
        << "Entropy should be non-negative";
}

TEST_F(BernoulliEdgeTest, LogProbaWithBoundaryProbabilities) {
    const auto proba = torch::tensor({{0.f, 1.f}});
    const auto action = torch::tensor({{1.f, 0.f}});

    const auto log_proba = bernoulli_log_proba(action, proba);

    ASSERT_TRUE(torch::all(torch::isfinite(log_proba)).item<bool>())
        << "Log-probability should be finite (clamped) even for impossible actions";
}

TEST_F(BernoulliEdgeTest, EntropyGradientFlowsThroughProbabilities) {
    const auto logits = torch::randn({4, 3}, torch::TensorOptions().requires_grad(true));
    const auto proba = torch::sigmoid(logits);

    const auto entropy = bernoulli_entropy(proba);
    const auto loss = entropy.sum();

    loss.backward();

    ASSERT_TRUE(logits.grad().defined()) << "Gradient should flow back through entropy";
    ASSERT_TRUE(torch::all(torch::isfinite(logits.grad())).item<bool>())
        << "Gradient should be finite";
}

TEST_F(BernoulliEdgeTest, SampleWithDeterministicProbabilities) {
    const auto proba = torch::cat({torch::zeros({4, 1}), torch::ones({4, 1})}, -1);

    const auto sample = bernoulli_sample(proba);

    ASSERT_TRUE(torch::allclose(sample.slice(-1, 0, 1), torch::zeros({4, 1})))
        << "p=0 should never engage";
    ASSERT_TRUE(torch::allclose(sample.slice(-1, 1, 2), torch::ones({4, 1})))
        << "p=1 should always engage";
}

TEST_F(BernoulliEdgeTest, MaxActionAtExactHalf) {
    const auto proba = torch::full({1, 2}, 0.5f);

    const auto action = bernoulli_max_action(proba);

    ASSERT_TRUE(torch::allclose(action, torch::zeros({1, 2})))
        << "p=0.5 exactly should not engage (strict > threshold)";
}
