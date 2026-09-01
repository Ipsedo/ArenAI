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
 * Target entropy
 */

TEST_F(ConstantTargetEntropyTest, StepIsANoOp) {
    ConstantTargetEntropy target(0.117f);

    target.step(1000000);

    ASSERT_NEAR(target.target_entropy().item<float>(), 0.117f, 1e-6f);
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

// inequality constraint H >= target: while the constraint is satisfied the multiplier is
// inactive — a sustained overshoot parks alpha at exactly 0, never below (the Lagrangian
// projection), so the bonus can never flip into a penalty
TEST_F(PidLagrangianAlphaParameterTest, SustainedOvershootParksAlphaAtZero) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 1);

    const auto target = torch::full({8, 1}, 0.7f);

    for (int i = 0; i < 1000; i++) pid.update(torch::full({8, 1}, 5.f), target);

    ASSERT_FLOAT_EQ(pid.alpha().item<float>(), 0.f);
}

// both the integral and the output are projected on [0, MAX_ALPHA], so even an absurd
// sustained error keeps alpha inside the feasible dual space
TEST_F(PidLagrangianAlphaParameterTest, StaysWithinZeroAndMax) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 1);

    const auto target = torch::full({8, 1}, 0.7f);

    for (int i = 0; i < 10000; i++) {
        pid.update(torch::full({8, 1}, -50.f), target);
        ASSERT_GE(pid.alpha().item<float>(), 0.f);
        ASSERT_LE(pid.alpha().item<float>(), 1.f + 1e-6f);
    }

    for (int i = 0; i < 10000; i++) {
        pid.update(torch::full({8, 1}, 50.f), target);
        ASSERT_GE(pid.alpha().item<float>(), 0.f);
        ASSERT_LE(pid.alpha().item<float>(), 1.f + 1e-6f);
    }
}

// the failure mode this controller replaces: the log-space integral used to saturate at
// log(1e-6), ~13 nats below zero, so alpha stayed pinned for thousands of updates after the
// error flipped sign. Here the integral saturates at the projection bound (0): the P term
// acts on the first update after the flip and the integral only has to climb from 0
TEST_F(PidLagrangianAlphaParameterTest, RecoversFromSaturationWithoutWindup) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 1);

    const auto target = torch::full({8, 1}, 0.7f);

    // long stretch above target: alpha parks at the zero floor
    for (int i = 0; i < 5000; i++) pid.update(torch::full({8, 1}, 5.f), target);
    ASSERT_FLOAT_EQ(pid.alpha().item<float>(), 0.f);

    // entropy now under target: the bonus has to come back within the run, not after it
    for (int i = 0; i < 1000; i++) pid.update(torch::full({8, 1}, 0.2f), target);
    ASSERT_GT(pid.alpha().item<float>(), 0.f);
}

// recovery time scales with 1 / (k_i * |error|), so a gain has to be scaled to the error
// range of its own head. The discrete target is the binary entropy of the fire probability
// -- 0.098 nat -- so its errors sit around 0.03, twenty times smaller than the continuous
// head's. Locks DISCRETE_ALPHA_K_I against that scale: with the gain it replaces, alpha
// stayed pinned at the floor for the whole of train_368
TEST_F(PidLagrangianAlphaParameterTest, DiscreteGainRecoversAtItsOwnErrorScale) {
    // DISCRETE_ALPHA_K_P / K_I / K_D and ALPHA_INITIAL, as set in ppo_trainer.cpp
    const PidLagrangianAlphaParameters pid(2e-1f, 1e-2f, 1.f, 1e-3f, 1);

    const auto target = torch::full({8, 1}, 0.098039f);

    // fire head still exploring: entropy above target, alpha parks at the zero floor
    for (int i = 0; i < 3000; i++) pid.update(torch::full({8, 1}, 0.693f), target);
    ASSERT_FLOAT_EQ(pid.alpha().item<float>(), 0.f);

    // entropy now under target: the bonus has to come back within the run, not after it
    for (int i = 0; i < 25000; i++) pid.update(torch::full({8, 1}, 0.068f), target);
    ASSERT_GT(pid.alpha().item<float>(), 1e-3f);
}

TEST_F(PidLagrangianAlphaParameterTest, OneAlphaPerAction) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 3);

    const auto entropy = torch::tensor({0.2f, 0.7f, 1.2f}).unsqueeze(0).repeat({8, 1});
    const auto target = torch::full({8, 3}, 0.7f);

    for (int i = 0; i < 100; i++) pid.update(entropy, target);

    const auto alpha = pid.alpha().squeeze(0);
    ASSERT_EQ(alpha.size(0), 3);
    ASSERT_GT(alpha[0].item<float>(), 0.f);
    ASSERT_GT(alpha[0].item<float>(), alpha[1].item<float>());
    ASSERT_GT(alpha[1].item<float>(), alpha[2].item<float>());
    ASSERT_FLOAT_EQ(alpha[2].item<float>(), 0.f);
}
