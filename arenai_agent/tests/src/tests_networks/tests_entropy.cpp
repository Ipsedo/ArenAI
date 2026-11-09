//
// Created by samuel on 30/06/2026.
//

#include <agents/entropy_alphas.h>
#include <networks/entropy.h>

#include <arenai_agent_tests/tests_networks/tests_entropy.h>

using namespace arenai;
using namespace arenai::agent;

/*
 * Learned alphas
 */

TEST_F(AlphaParametersTest, StartsAtInitialAlpha) {
    const AlphaParameters alphas(1e-3f, 3);

    ASSERT_TRUE(torch::allclose(alphas.alpha(), torch::full({3}, 1e-3f)));
}

// descending the dual loss raises the alpha of an action whose entropy sits under
// its target and lowers the one of an action above it, each on its own
TEST_F(AlphaParametersTest, DualStepMovesEachAlphaWithItsOwnError) {
    AlphaParameters alphas(1e-3f, 2);
    torch::optim::Adam optim(alphas.parameters(), 1e-2);

    const auto before = alphas.alpha().detach().clone();

    // H - H_target: the first action is under its target, the second above
    const auto error = torch::tensor({-1.f, 1.f});
    for (int i = 0; i < 10; i++) {
        optim.zero_grad();
        alphas.loss(error).backward();
        optim.step();
    }

    const auto after = alphas.alpha().detach();
    ASSERT_GT(after[0].item<float>(), before[0].item<float>());
    ASSERT_LT(after[1].item<float>(), before[1].item<float>());
}

TEST_F(AlphaParametersTest, SustainedOvershootParksAlphaAtTheFloor) {
    AlphaParameters alphas(1e-2f, 1);
    torch::optim::Adam optim(alphas.parameters(), 1e-2);

    for (int i = 0; i < 1000; i++) {
        optim.zero_grad();
        alphas.loss(torch::tensor({10.f})).backward();
        optim.step();
        alphas.clamp();
        ASSERT_GE(alphas.alpha().item<float>(), AlphaParameters::MIN_ALPHA * (1.f - 1e-5f));
    }

    ASSERT_NEAR(alphas.alpha().item<float>(), AlphaParameters::MIN_ALPHA, 1e-6f);
}

TEST_F(AlphaParametersTest, SustainedUndershootParksAlphaAtTheCeiling) {
    AlphaParameters alphas(1e-2f, 1);
    torch::optim::Adam optim(alphas.parameters(), 1e-1);

    for (int i = 0; i < 1000; i++) {
        optim.zero_grad();
        alphas.loss(torch::tensor({-10.f})).backward();
        optim.step();
        alphas.clamp();
        ASSERT_LE(alphas.alpha().item<float>(), AlphaParameters::MAX_ALPHA * (1.f + 1e-5f));
    }

    ASSERT_NEAR(alphas.alpha().item<float>(), AlphaParameters::MAX_ALPHA, 1e-5f);
}

// the floor removes the windup: after a long overshoot, a few steps of undershoot (the
// span of Adam's momentum) are enough to leave it, where an unbounded log_alpha would first
// have to climb all the way back
TEST_F(AlphaParametersTest, AlphaLeavesTheFloorWithinAFewUndershootSteps) {
    AlphaParameters alphas(1e-2f, 1);
    torch::optim::Adam optim(alphas.parameters(), 1e-2);

    for (int i = 0; i < 1000; i++) {
        optim.zero_grad();
        alphas.loss(torch::tensor({10.f})).backward();
        optim.step();
        alphas.clamp();
    }
    ASSERT_NEAR(alphas.alpha().item<float>(), AlphaParameters::MIN_ALPHA, 1e-6f);

    for (int i = 0; i < 50; i++) {
        optim.zero_grad();
        alphas.loss(torch::tensor({-10.f})).backward();
        optim.step();
        alphas.clamp();
    }
    ASSERT_GT(alphas.alpha().item<float>(), AlphaParameters::MIN_ALPHA * 1.2f);
}

TEST_F(AlphaParametersTest, InitialAlphaIsProjectedOnTheBounds) {
    const AlphaParameters below(1e-6f, 1);
    const AlphaParameters above(10.f, 1);

    ASSERT_NEAR(below.alpha().item<float>(), AlphaParameters::MIN_ALPHA, 1e-6f);
    ASSERT_NEAR(above.alpha().item<float>(), AlphaParameters::MAX_ALPHA, 1e-5f);
}

/*
 * Per-action regulator
 */

TEST_F(EntropyAlphasTest, OneMetricPerActionNamedAfterIt) {
    const EntropyAlphas alphas(4, 2, 1e-3f, 3e-4f, -1.f, 0.05f);

    const auto metrics = alphas.get_metrics();

    ASSERT_EQ(metrics.size(), 6);
    ASSERT_EQ(metrics[0]->get_name(), "alpha_dir_x");
    ASSERT_EQ(metrics[3]->get_name(), "alpha_canon_y");
    ASSERT_EQ(metrics[4]->get_name(), "alpha_fire");
    ASSERT_EQ(metrics[5]->get_name(), "alpha_zoom");
    for (const auto &metric: metrics) ASSERT_NEAR(metric->compute_metric(), 1e-3f, 1e-7f);
}

TEST_F(EntropyAlphasTest, EntropyUnderTargetRaisesAlphaAboveLowersIt) {
    const EntropyAlphas alphas(2, 2, 1e-2f, 1e-2f, -1.f, 0.05f);

    // continuous: first action under -1, second above; discrete: fire under log 2 / 2 ≈ 0.2,
    // zoom at the maximum entropy
    const auto continuous_entropy = torch::tensor({{-2.f, 0.f}}).expand({8, 2});
    const auto discrete_entropy = torch::tensor({{0.05f, 0.69f}}).expand({8, 2});
    const auto mask = torch::ones({8, 2});

    const auto continuous_before = alphas.continuous().detach().clone();
    const auto discrete_before = alphas.discrete().detach().clone();

    for (int i = 0; i < 10; i++) {
        alphas.zero_grad();
        alphas.backward(continuous_entropy, discrete_entropy, mask, 1.f);
        alphas.step();
    }

    ASSERT_GT(alphas.continuous()[0].item<float>(), continuous_before[0].item<float>());
    ASSERT_LT(alphas.continuous()[1].item<float>(), continuous_before[1].item<float>());
    ASSERT_GT(alphas.discrete()[0].item<float>(), discrete_before[0].item<float>());
    ASSERT_LT(alphas.discrete()[1].item<float>(), discrete_before[1].item<float>());
}

// a discrete action that never had an effect brings no row to its error: its alpha
// stays put while the others move
TEST_F(EntropyAlphasTest, MaskedDiscreteActionKeepsItsAlpha) {
    const EntropyAlphas alphas(1, 2, 1e-3f, 1e-2f, -1.f, 0.05f);

    const auto continuous_entropy = torch::full({8, 1}, -3.f);
    const auto discrete_entropy = torch::full({8, 2}, 0.f);
    const auto mask = torch::tensor({{0.f, 1.f}}).expand({8, 2});

    for (int i = 0; i < 10; i++) {
        alphas.zero_grad();
        alphas.backward(continuous_entropy, discrete_entropy, mask, 1.f);
        alphas.step();
    }

    ASSERT_FLOAT_EQ(alphas.discrete()[0].item<float>(), 1e-3f);
    ASSERT_GT(alphas.discrete()[1].item<float>(), 1e-3f);
    ASSERT_GT(alphas.continuous()[0].item<float>(), 1e-3f);
}

// the step projects every alpha back on its bounds: a sustained overshoot parks them at the
// floor instead of driving log_alpha away
TEST_F(EntropyAlphasTest, StepKeepsEveryAlphaAboveTheFloor) {
    const EntropyAlphas alphas(2, 2, 1e-2f, 1e-1f, -1.f, 0.05f);

    const auto continuous_entropy = torch::full({8, 2}, 5.f);
    const auto discrete_entropy = torch::full({8, 2}, 0.69f);
    const auto mask = torch::ones({8, 2});

    for (int i = 0; i < 200; i++) {
        alphas.zero_grad();
        alphas.backward(continuous_entropy, discrete_entropy, mask, 1.f);
        alphas.step();
    }

    ASSERT_TRUE(torch::allclose(
        alphas.continuous(), torch::full({2}, AlphaParameters::MIN_ALPHA), 1e-4, 1e-7));
    ASSERT_TRUE(torch::allclose(
        alphas.discrete(), torch::full({2}, AlphaParameters::MIN_ALPHA), 1e-4, 1e-7));
    for (const auto &metric: alphas.get_metrics())
        ASSERT_NEAR(metric->compute_metric(), AlphaParameters::MIN_ALPHA, 1e-6f);
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
