//
// Created by claude on 01/07/2026.
//

#include <distributions/beta_law.h>

#include <arenai_agent_tests/tests_distributions/tests_beta_law_edge.h>

using namespace arenai;
using namespace arenai::agent;

// ========================================================================
// Asymmetric parameter edge cases
// ========================================================================

TEST_F(BetaLawEdgeTest, ModeNearUpperBound) {
    const auto mode = torch::ones({50}) * 0.999f;
    const auto concentration = torch::ones({50}) * 100.0f;

    const auto samples = beta_law_sample(mode, concentration);
    const auto log_p = beta_law_log_proba(samples, mode, concentration);
    const auto entropy = beta_law_entropy(mode, concentration);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>())
        << "Samples should be finite with mode near 1";
    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>())
        << "Log-proba should be finite with mode near 1";
    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>())
        << "Entropy should be finite with mode near 1";
}

TEST_F(BetaLawEdgeTest, ModeNearLowerBound) {
    const auto mode = torch::ones({50}) * 0.001f;
    const auto concentration = torch::ones({50}) * 100.0f;

    const auto samples = beta_law_sample(mode, concentration);
    const auto log_p = beta_law_log_proba(samples, mode, concentration);
    const auto entropy = beta_law_entropy(mode, concentration);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>())
        << "Samples should be finite with mode near 0";
    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>())
        << "Log-proba should be finite with mode near 0";
    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>())
        << "Entropy should be finite with mode near 0";
}

TEST_F(BetaLawEdgeTest, ZeroParamsHandledGracefully) {
    const auto mode = torch::zeros({10});
    const auto concentration = torch::zeros({10});

    const auto samples = beta_law_sample(mode, concentration);
    const auto entropy = beta_law_entropy(mode, concentration);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>())
        << "Samples should be finite with zero params (clamped to EPSILON)";
    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>())
        << "Entropy should be finite with zero params";
}

TEST_F(BetaLawEdgeTest, NegativeParamsHandledGracefully) {
    const auto mode = torch::ones({10}) * -1.0f;
    const auto concentration = torch::ones({10}) * -1.0f;

    const auto samples = beta_law_sample(mode, concentration);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>())
        << "Samples should be finite with negative params (clamped to EPSILON)";
    ASSERT_TRUE(torch::all(torch::logical_and(torch::ge(samples, -1.f), torch::le(samples, 1.f)))
                    .item<bool>());
}

TEST_F(BetaLawEdgeTest, LogProbAtBoundaryValues) {
    const auto mode = torch::ones({10}) * 0.5f;
    const auto concentration = torch::ones({10}) * 6.0f;
    const auto x_near_minus1 = torch::ones({10}) * -0.999f;
    const auto x_near_plus1 = torch::ones({10}) * 0.999f;

    const auto log_p_lo = beta_law_log_proba(x_near_minus1, mode, concentration);
    const auto log_p_hi = beta_law_log_proba(x_near_plus1, mode, concentration);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p_lo)).item<bool>())
        << "Log-proba near -1 boundary should be finite";
    ASSERT_TRUE(torch::all(torch::isfinite(log_p_hi)).item<bool>())
        << "Log-proba near +1 boundary should be finite";
}

TEST_F(BetaLawEdgeTest, LogProbAtExactBoundaryValues) {
    const auto mode = torch::ones({10}) * 0.5f;
    const auto concentration = torch::ones({10}) * 6.0f;
    const auto x_minus1 = torch::ones({10}) * -1.0f;
    const auto x_plus1 = torch::ones({10}) * 1.0f;

    const auto log_p_lo = beta_law_log_proba(x_minus1, mode, concentration);
    const auto log_p_hi = beta_law_log_proba(x_plus1, mode, concentration);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p_lo)).item<bool>())
        << "Log-proba at exact -1 boundary should be finite (clamped)";
    ASSERT_TRUE(torch::all(torch::isfinite(log_p_hi)).item<bool>())
        << "Log-proba at exact +1 boundary should be finite (clamped)";
}

// ========================================================================
// Gradient flow tests
// ========================================================================

TEST_F(BetaLawGradientTest, LogProbaGradientFlowsThroughMode) {
    const auto mode = torch::full({5}, 0.4f, torch::TensorOptions().requires_grad(true));
    const auto concentration = torch::ones({5}) * 6.0f;
    const auto x = torch::tensor({0.0f, 0.2f, -0.3f, 0.5f, -0.1f});

    const auto log_p = beta_law_log_proba(x, mode, concentration);
    const auto loss = log_p.sum();

    loss.backward();

    ASSERT_TRUE(mode.grad().defined()) << "Gradient should flow back to mode";
    ASSERT_TRUE(torch::all(torch::isfinite(mode.grad())).item<bool>())
        << "Gradient w.r.t. mode should be finite";
}

TEST_F(BetaLawGradientTest, LogProbaGradientFlowsThroughConcentration) {
    const auto mode = torch::ones({5}) * 0.4f;
    const auto concentration = torch::full({5}, 6.0f, torch::TensorOptions().requires_grad(true));
    const auto x = torch::tensor({0.0f, 0.2f, -0.3f, 0.5f, -0.1f});

    const auto log_p = beta_law_log_proba(x, mode, concentration);
    const auto loss = log_p.sum();

    loss.backward();

    ASSERT_TRUE(concentration.grad().defined()) << "Gradient should flow back to concentration";
    ASSERT_TRUE(torch::all(torch::isfinite(concentration.grad())).item<bool>())
        << "Gradient w.r.t. concentration should be finite";
}

TEST_F(BetaLawGradientTest, EntropyGradientFlowsThroughMode) {
    const auto mode = torch::full({5}, 0.4f, torch::TensorOptions().requires_grad(true));
    const auto concentration = torch::ones({5}) * 6.0f;

    const auto entropy = beta_law_entropy(mode, concentration);
    const auto loss = entropy.sum();

    loss.backward();

    ASSERT_TRUE(mode.grad().defined()) << "Entropy gradient should flow back to mode";
    ASSERT_TRUE(torch::all(torch::isfinite(mode.grad())).item<bool>())
        << "Entropy gradient w.r.t. mode should be finite";
}

TEST_F(BetaLawGradientTest, EntropyGradientFlowsThroughConcentration) {
    const auto mode = torch::ones({5}) * 0.4f;
    const auto concentration = torch::full({5}, 6.0f, torch::TensorOptions().requires_grad(true));

    const auto entropy = beta_law_entropy(mode, concentration);
    const auto loss = entropy.sum();

    loss.backward();

    ASSERT_TRUE(concentration.grad().defined())
        << "Entropy gradient should flow back to concentration";
    ASSERT_TRUE(torch::all(torch::isfinite(concentration.grad())).item<bool>())
        << "Entropy gradient w.r.t. concentration should be finite";
}
