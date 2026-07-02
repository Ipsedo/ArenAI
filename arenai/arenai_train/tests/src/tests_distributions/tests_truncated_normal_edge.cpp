//
// Created by claude on 01/07/2026.
//

#include <distributions/truncated_normal.h>

#include <arenai_core/constants.h>
#include <arenai_train_tests/tests_distributions/tests_truncated_normal_edge.h>

using namespace arenai;
using namespace arenai::train;

TEST_F(TruncatedNormalEdgeTest, SampleWithVerySigmaSmallProducesFinite) {
    const auto mu = torch::zeros({100});
    const auto sigma = torch::ones({100}) * 1e-10f;

    const auto samples = truncated_normal_sample(mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>())
        << "Sample should be finite with very small sigma";
}

TEST_F(TruncatedNormalEdgeTest, EntropyWithVerySigmaSmallProducesFinite) {
    const auto mu = torch::zeros({100});
    const auto sigma = torch::ones({100}) * 1e-10f;

    const auto entropy = truncated_normal_entropy(mu, sigma);

    ASSERT_TRUE(torch::all(torch::isfinite(entropy)).item<bool>())
        << "Entropy should be finite with very small sigma";
}

TEST_F(TruncatedNormalEdgeTest, SampleWithMuAtBoundary) {
    const auto mu = torch::ones({100});
    const auto sigma = torch::ones({100}) * 0.1f;

    const auto samples = truncated_normal_sample(mu, sigma, -1.f, 1.f);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>());
    ASSERT_TRUE(torch::all(torch::logical_and(torch::ge(samples, -1.f), torch::le(samples, 1.f)))
                    .item<bool>());
}

TEST_F(TruncatedNormalEdgeTest, SampleWithMuOutsideBounds) {
    const auto mu = torch::ones({100}) * 5.f;
    const auto sigma = torch::ones({100}) * 0.5f;

    const auto samples = truncated_normal_sample(mu, sigma, -1.f, 1.f);

    ASSERT_TRUE(torch::all(torch::isfinite(samples)).item<bool>());
    ASSERT_TRUE(torch::all(torch::logical_and(torch::ge(samples, -1.f), torch::le(samples, 1.f)))
                    .item<bool>())
        << "Samples must stay within bounds even when mu is outside";
}

TEST_F(TruncatedNormalEdgeTest, LogPdfAndPdfConsistent) {
    const auto mu = torch::randn({50});
    const auto sigma = torch::rand({50}) + 0.1f;
    const auto x = torch::clamp(mu + torch::randn({50}) * 0.1f, -0.99f, 0.99f);

    const auto log_pdf = truncated_normal_log_pdf(x, mu, sigma);
    const auto pdf = truncated_normal_pdf(x, mu, sigma);

    const auto log_pdf_from_pdf = torch::log(torch::clamp_min(pdf, 1e-30f));

    ASSERT_TRUE(torch::allclose(log_pdf, log_pdf_from_pdf, 1e-2, 1e-5))
        << "log_pdf and log(pdf) should be consistent";
}

// ========================================================================
// Gradient flow tests — critical for SAC training
// ========================================================================

TEST_F(TruncatedNormalGradientTest, LogPdfGradientFlowsThroughMu) {
    auto mu = torch::zeros({5}, torch::TensorOptions().requires_grad(true));
    const auto sigma = torch::ones({5}) * 0.5f;
    const auto x = torch::tensor({0.1f, -0.2f, 0.3f, -0.1f, 0.0f});

    const auto log_pdf = truncated_normal_log_pdf(x, mu, sigma);
    const auto loss = log_pdf.sum();

    loss.backward();

    ASSERT_TRUE(mu.grad().defined()) << "Gradient should flow back to mu";
    ASSERT_TRUE(torch::all(torch::isfinite(mu.grad())).item<bool>())
        << "Gradient w.r.t. mu should be finite";
    ASSERT_FALSE(torch::allclose(mu.grad(), torch::zeros_like(mu.grad())))
        << "Gradient w.r.t. mu should be non-zero";
}

TEST_F(TruncatedNormalGradientTest, LogPdfGradientFlowsThroughSigma) {
    const auto mu = torch::zeros({5});
    auto sigma = torch::full({5}, 0.5f, torch::TensorOptions().requires_grad(true));
    const auto x = torch::tensor({0.1f, -0.2f, 0.3f, -0.1f, 0.0f});

    const auto log_pdf = truncated_normal_log_pdf(x, mu, sigma);
    const auto loss = log_pdf.sum();

    loss.backward();

    ASSERT_TRUE(sigma.grad().defined()) << "Gradient should flow back to sigma";
    ASSERT_TRUE(torch::all(torch::isfinite(sigma.grad())).item<bool>())
        << "Gradient w.r.t. sigma should be finite";
}

TEST_F(TruncatedNormalGradientTest, EntropyGradientFlowsThroughMu) {
    auto mu = torch::zeros({5}, torch::TensorOptions().requires_grad(true));
    const auto sigma = torch::ones({5}) * 0.5f;

    const auto entropy = truncated_normal_entropy(mu, sigma);
    const auto loss = entropy.sum();

    loss.backward();

    ASSERT_TRUE(mu.grad().defined()) << "Gradient should flow back to mu";
    ASSERT_TRUE(torch::all(torch::isfinite(mu.grad())).item<bool>())
        << "Gradient w.r.t. mu should be finite";
}

TEST_F(TruncatedNormalGradientTest, EntropyGradientFlowsThroughSigma) {
    const auto mu = torch::zeros({5});
    auto sigma = torch::full({5}, 0.5f, torch::TensorOptions().requires_grad(true));

    const auto entropy = truncated_normal_entropy(mu, sigma);
    const auto loss = entropy.sum();

    loss.backward();

    ASSERT_TRUE(sigma.grad().defined()) << "Gradient should flow back to sigma";
    ASSERT_TRUE(torch::all(torch::isfinite(sigma.grad())).item<bool>())
        << "Gradient w.r.t. sigma should be finite";
}
