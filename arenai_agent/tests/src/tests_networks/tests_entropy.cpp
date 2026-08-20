//
// Created by samuel on 30/06/2026.
//

#include <networks/entropy.h>

#include <arenai_agent_tests/tests_networks/tests_entropy.h>

using namespace arenai;
using namespace arenai::agent;

TEST_F(AlphaParameterTest, AlphaAlwaysPositive) {
    for (const float init: {0.01f, 0.1f, 1.0f, 10.0f}) {
        AlphaParameters param(init, 1);
        ASSERT_GT(param.alpha().item<float>(), 0.0f)
            << "alpha should be positive for initial_alpha=" << init;
    }
}

TEST_F(AlphaParameterTest, InitialValueMatchesInput) {
    constexpr float initial = 0.2f;
    AlphaParameters param(initial, 1);

    ASSERT_NEAR(param.alpha().item<float>(), initial, 1e-6f);
}

TEST_F(AlphaParameterTest, LogAlphaRequiresGrad) {
    AlphaParameters param(1.0f, 1);

    ASSERT_TRUE(param.log_alpha().requires_grad());
}

TEST_F(AlphaParameterTest, LogAlphaConsistentWithAlpha) {
    AlphaParameters param(0.5f, 1);

    const auto log_a = param.log_alpha().item<float>();
    const auto a = param.alpha().item<float>();

    ASSERT_NEAR(std::exp(log_a), a, 1e-6f);
}

/*
 * PID Lagrangian
 */

TEST_F(PidLagrangianAlphaParameterTest, StartsAtInitialAlpha) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 1);

    ASSERT_NEAR(pid.alpha().item<float>(), 1e-3f, 1e-7f);
}

TEST_F(PidLagrangianAlphaParameterTest, NullErrorKeepsAlpha) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 1);

    const auto entropy = torch::full({8, 1}, 0.5f);

    for (int i = 0; i < 100; i++) pid.update(entropy, entropy);

    ASSERT_NEAR(pid.alpha().item<float>(), 1e-3f, 1e-6f);
}

TEST_F(PidLagrangianAlphaParameterTest, EntropyBelowTargetRaisesAlpha) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 1);

    const auto entropy = torch::full({8, 1}, 0.2f);
    const auto target = torch::full({8, 1}, 0.7f);

    const auto before = pid.alpha().item<float>();
    for (int i = 0; i < 10; i++) pid.update(entropy, target);

    ASSERT_GT(pid.alpha().item<float>(), before);
}

// a multiplier never goes negative, whichever side the error sits on: the entropy bonus can
// be switched off, never turned into a penalty. Running on log(alpha) gives this for free,
// and the bounded integral keeps the excursion finite on both sides
TEST_F(PidLagrangianAlphaParameterTest, StaysPositiveAndFinite) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 1);

    const auto target = torch::full({8, 1}, 0.7f);

    for (int i = 0; i < 10000; i++) {
        pid.update(torch::full({8, 1}, -5.f), target);
        const auto alpha = pid.alpha().item<float>();
        ASSERT_GT(alpha, 0.f);
        ASSERT_LT(alpha, 10.f);
    }

    for (int i = 0; i < 10000; i++) {
        pid.update(torch::full({8, 1}, 5.f), target);
        const auto alpha = pid.alpha().item<float>();
        ASSERT_GT(alpha, 0.f);
        ASSERT_LT(alpha, 10.f);
    }
}

// the failure mode this controller replaces: a sustained negative error used to drive the
// integral arbitrarily low, so alpha could not come back once the error flipped sign
TEST_F(PidLagrangianAlphaParameterTest, RecoversFromSaturationWithoutWindup) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 1);

    const auto target = torch::full({8, 1}, 0.7f);

    // long stretch above target: alpha bottoms out
    for (int i = 0; i < 5000; i++) pid.update(torch::full({8, 1}, 5.f), target);
    ASSERT_LT(pid.alpha().item<float>(), 1e-7f);

    // the integral is bounded, so coming back up takes (log range) / (k_i * error) updates
    // and not the unbounded time an unclamped integral would need
    for (int i = 0; i < 5000; i++) pid.update(torch::full({8, 1}, 0.2f), target);
    ASSERT_GT(pid.alpha().item<float>(), 1e-3f);
}

TEST_F(PidLagrangianAlphaParameterTest, OneAlphaPerAction) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 3);

    const auto entropy = torch::tensor({0.2f, 0.7f, 1.2f}).unsqueeze(0).repeat({8, 1});
    const auto target = torch::full({8, 3}, 0.7f);

    for (int i = 0; i < 100; i++) pid.update(entropy, target);

    const auto alpha = pid.alpha().squeeze(0);
    ASSERT_EQ(alpha.size(0), 3);
    ASSERT_GT(alpha[0].item<float>(), alpha[1].item<float>());
    ASSERT_GT(alpha[1].item<float>(), alpha[2].item<float>());
}
