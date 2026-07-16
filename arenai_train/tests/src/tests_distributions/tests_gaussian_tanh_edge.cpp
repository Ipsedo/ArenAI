//
// Created by claude on 01/07/2026.
//

#include <distributions/gaussian_tanh.h>

#include <arenai_train_tests/tests_distributions/tests_gaussian_tanh_edge.h>

using namespace arenai;
using namespace arenai::train;

// ========================================================================
// Edge cases: extreme values
// ========================================================================

TEST_F(GaussianTanhEdgeTest, LogPdfWithLargeUStaysFinite) {
    const auto mu = torch::zeros({10});
    const auto sigma = torch::ones({10}) * 0.5f;
    const auto u = torch::ones({10}) * 5.0f;

    const auto log_p = gaussian_tanh_log_pdf(u, mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>())
        << "log_pdf should be finite even with u=5 (tanh(5) ≈ 0.9999)";
}

TEST_F(GaussianTanhEdgeTest, LogPdfWithVeryLargeUStaysFinite) {
    const auto mu = torch::zeros({10});
    const auto sigma = torch::ones({10}) * 0.5f;
    const auto u = torch::ones({10}) * 20.0f;

    const auto log_p = gaussian_tanh_log_pdf(u, mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>())
        << "log_pdf should be finite even with u=20 (tanh(20) ≈ 1.0)";
}

TEST_F(GaussianTanhEdgeTest, LogPdfWithNegativeLargeUStaysFinite) {
    const auto mu = torch::zeros({10});
    const auto sigma = torch::ones({10}) * 0.5f;
    const auto u = torch::ones({10}) * -20.0f;

    const auto log_p = gaussian_tanh_log_pdf(u, mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>())
        << "log_pdf should be finite with large negative u";
}

TEST_F(GaussianTanhEdgeTest, LogPdfWithZeroSigma) {
    const auto mu = torch::zeros({10});
    const auto sigma = torch::zeros({10});
    const auto u = torch::randn({10}) * 0.1f;

    const auto log_p = gaussian_tanh_log_pdf(u, mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(log_p)).item<bool>())
        << "log_pdf should handle zero sigma via clamping";
}

TEST_F(GaussianTanhEdgeTest, SampleWithVeryLargeMu) {
    const auto mu = torch::ones({100}) * 100.f;
    const auto sigma = torch::ones({100}) * 0.1f;

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(action)).item<bool>())
        << "Action should be finite with large mu";
    ASSERT_TRUE(torch::all(torch::logical_and(torch::ge(action, -1.0f), torch::le(action, 1.0f)))
                    .item<bool>())
        << "Action should still be bounded by tanh";
}

TEST_F(GaussianTanhEdgeTest, SampleWithZeroSigma) {
    const auto mu = torch::zeros({100});
    const auto sigma = torch::zeros({100});

    const auto [action, u] = gaussian_tanh_sample(mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(action)).item<bool>())
        << "Action should be finite with zero sigma (clamped)";
}

// ========================================================================
// Gradient flow tests
// ========================================================================

TEST_F(GaussianTanhGradientTest, LogPdfGradientFlowsThroughMu) {
    auto mu = torch::zeros({5}, torch::TensorOptions().requires_grad(true));
    const auto sigma = torch::ones({5}) * 0.5f;
    const auto u = torch::tensor({0.1f, -0.2f, 0.3f, -0.1f, 0.0f});

    const auto log_p = gaussian_tanh_log_pdf(u, mu, sigma);
    const auto loss = log_p.sum();

    loss.backward();

    ASSERT_TRUE(mu.grad().defined()) << "Gradient should flow back to mu";
    ASSERT_TRUE(torch::all(torch::isfinite(mu.grad())).item<bool>())
        << "Gradient w.r.t. mu should be finite";
    ASSERT_FALSE(torch::allclose(mu.grad(), torch::zeros_like(mu.grad())))
        << "Gradient w.r.t. mu should be non-zero";
}

TEST_F(GaussianTanhGradientTest, LogPdfGradientFlowsThroughSigma) {
    const auto mu = torch::zeros({5});
    auto sigma = torch::full({5}, 0.5f, torch::TensorOptions().requires_grad(true));
    const auto u = torch::tensor({0.1f, -0.2f, 0.3f, -0.1f, 0.0f});

    const auto log_p = gaussian_tanh_log_pdf(u, mu, sigma);
    const auto loss = log_p.sum();

    loss.backward();

    ASSERT_TRUE(sigma.grad().defined()) << "Gradient should flow back to sigma";
    ASSERT_TRUE(torch::all(torch::isfinite(sigma.grad())).item<bool>())
        << "Gradient w.r.t. sigma should be finite";
}

TEST_F(GaussianTanhGradientTest, LogPdfGradientFiniteWithLargeU) {
    auto mu = torch::zeros({5}, torch::TensorOptions().requires_grad(true));
    const auto sigma = torch::ones({5}) * 0.5f;
    const auto u = torch::ones({5}) * 10.0f;

    const auto log_p = gaussian_tanh_log_pdf(u, mu, sigma);
    const auto loss = log_p.sum();

    loss.backward();

    ASSERT_TRUE(torch::all(torch::isfinite(mu.grad())).item<bool>())
        << "Gradient should be finite even with large u (near tanh saturation)";
}
