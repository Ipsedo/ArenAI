//
// Created by claude on 01/07/2026.
//

#include <distributions/beta_law.h>

#include <arenai_train_tests/tests_distributions/tests_beta_law_edge.h>

using namespace arenai;
using namespace arenai::train;

// ========================================================================
// Asymmetric parameter edge cases
// ========================================================================

TEST_F(BetaLawEdgeTest, VeryAsymmetricParamsAlphaLarge) {
    const auto alpha = torch::ones({50}) * 100.0f;
    const auto beta = torch::ones({50}) * 0.01f;

    const auto samples = beta_law_sample(alpha, beta);
    const auto log_p = beta_law_log_proba(samples, alpha, beta);
    const auto entropy = beta_law_entropy(alpha, beta);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>())
        << "Samples should be finite with alpha>>beta";
    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>())
        << "Log-proba should be finite with alpha>>beta";
    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>())
        << "Entropy should be finite with alpha>>beta";
}

TEST_F(BetaLawEdgeTest, VeryAsymmetricParamsBetaLarge) {
    const auto alpha = torch::ones({50}) * 0.01f;
    const auto beta = torch::ones({50}) * 100.0f;

    const auto samples = beta_law_sample(alpha, beta);
    const auto log_p = beta_law_log_proba(samples, alpha, beta);
    const auto entropy = beta_law_entropy(alpha, beta);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>())
        << "Samples should be finite with beta>>alpha";
    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>())
        << "Log-proba should be finite with beta>>alpha";
    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>())
        << "Entropy should be finite with beta>>alpha";
}

TEST_F(BetaLawEdgeTest, ZeroParamsHandledGracefully) {
    const auto alpha = torch::zeros({10});
    const auto beta = torch::zeros({10});

    const auto samples = beta_law_sample(alpha, beta);
    const auto entropy = beta_law_entropy(alpha, beta);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>())
        << "Samples should be finite with zero params (clamped to EPSILON)";
    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>())
        << "Entropy should be finite with zero params";
}

TEST_F(BetaLawEdgeTest, NegativeParamsHandledGracefully) {
    const auto alpha = torch::ones({10}) * -1.0f;
    const auto beta = torch::ones({10}) * -1.0f;

    const auto samples = beta_law_sample(alpha, beta);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>())
        << "Samples should be finite with negative params (clamped to EPSILON)";
    ASSERT_TRUE(torch::all(torch::logical_and(torch::ge(samples, -1.f), torch::le(samples, 1.f)))
                    .item<bool>());
}

TEST_F(BetaLawEdgeTest, LogProbAtBoundaryValues) {
    const auto alpha = torch::ones({10}) * 2.0f;
    const auto beta = torch::ones({10}) * 2.0f;
    const auto x_near_minus1 = torch::ones({10}) * -0.999f;
    const auto x_near_plus1 = torch::ones({10}) * 0.999f;

    const auto log_p_lo = beta_law_log_proba(x_near_minus1, alpha, beta);
    const auto log_p_hi = beta_law_log_proba(x_near_plus1, alpha, beta);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p_lo)).item<bool>())
        << "Log-proba near -1 boundary should be finite";
    ASSERT_TRUE(torch::all(torch::isfinite(log_p_hi)).item<bool>())
        << "Log-proba near +1 boundary should be finite";
}

TEST_F(BetaLawEdgeTest, LogProbAtExactBoundaryValues) {
    const auto alpha = torch::ones({10}) * 2.0f;
    const auto beta = torch::ones({10}) * 2.0f;
    const auto x_minus1 = torch::ones({10}) * -1.0f;
    const auto x_plus1 = torch::ones({10}) * 1.0f;

    const auto log_p_lo = beta_law_log_proba(x_minus1, alpha, beta);
    const auto log_p_hi = beta_law_log_proba(x_plus1, alpha, beta);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p_lo)).item<bool>())
        << "Log-proba at exact -1 boundary should be finite (clamped)";
    ASSERT_TRUE(torch::all(torch::isfinite(log_p_hi)).item<bool>())
        << "Log-proba at exact +1 boundary should be finite (clamped)";
}

// ========================================================================
// Gradient flow tests
// ========================================================================

TEST_F(BetaLawGradientTest, LogProbaGradientFlowsThroughAlpha) {
    auto alpha = torch::full({5}, 2.0f, torch::TensorOptions().requires_grad(true));
    const auto beta = torch::ones({5}) * 3.0f;
    const auto x = torch::tensor({0.0f, 0.2f, -0.3f, 0.5f, -0.1f});

    const auto log_p = beta_law_log_proba(x, alpha, beta);
    const auto loss = log_p.sum();

    loss.backward();

    ASSERT_TRUE(alpha.grad().defined()) << "Gradient should flow back to alpha";
    ASSERT_TRUE(torch::all(torch::isfinite(alpha.grad())).item<bool>())
        << "Gradient w.r.t. alpha should be finite";
}

TEST_F(BetaLawGradientTest, LogProbaGradientFlowsThroughBeta) {
    const auto alpha = torch::ones({5}) * 2.0f;
    auto beta = torch::full({5}, 3.0f, torch::TensorOptions().requires_grad(true));
    const auto x = torch::tensor({0.0f, 0.2f, -0.3f, 0.5f, -0.1f});

    const auto log_p = beta_law_log_proba(x, alpha, beta);
    const auto loss = log_p.sum();

    loss.backward();

    ASSERT_TRUE(beta.grad().defined()) << "Gradient should flow back to beta";
    ASSERT_TRUE(torch::all(torch::isfinite(beta.grad())).item<bool>())
        << "Gradient w.r.t. beta should be finite";
}

TEST_F(BetaLawGradientTest, EntropyGradientFlowsThroughAlpha) {
    auto alpha = torch::full({5}, 2.0f, torch::TensorOptions().requires_grad(true));
    const auto beta = torch::ones({5}) * 3.0f;

    const auto entropy = beta_law_entropy(alpha, beta);
    const auto loss = entropy.sum();

    loss.backward();

    ASSERT_TRUE(alpha.grad().defined()) << "Entropy gradient should flow back to alpha";
    ASSERT_TRUE(torch::all(torch::isfinite(alpha.grad())).item<bool>())
        << "Entropy gradient w.r.t. alpha should be finite";
}

TEST_F(BetaLawGradientTest, EntropyGradientFlowsThroughBeta) {
    const auto alpha = torch::ones({5}) * 2.0f;
    auto beta = torch::full({5}, 3.0f, torch::TensorOptions().requires_grad(true));

    const auto entropy = beta_law_entropy(alpha, beta);
    const auto loss = entropy.sum();

    loss.backward();

    ASSERT_TRUE(beta.grad().defined()) << "Entropy gradient should flow back to beta";
    ASSERT_TRUE(torch::all(torch::isfinite(beta.grad())).item<bool>())
        << "Entropy gradient w.r.t. beta should be finite";
}
