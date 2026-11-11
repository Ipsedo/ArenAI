#include <agents/entropy_alphas.h>
#include <networks/entropy.h>

#include <arenai_agent_tests/tests_networks/tests_entropy.h>

using namespace arenai;
using namespace arenai::agent;

TEST_F(AlphaParametersTest, StartsAtInitialAlpha) {
    const AlphaParameters alphas(1e-3f, 3);

    ASSERT_TRUE(torch::allclose(alphas.alpha(), torch::full({3}, 1e-3f)));
}

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

TEST_F(EntropyAlphasTest, OneMetricPerActionNamedAfterIt) {
    const EntropyAlphas alphas(5, 2, 1e-3f, 3e-4f, std::vector(5, -1.f), 0.05f);

    const auto metrics = alphas.get_metrics();

    ASSERT_EQ(metrics.size(), 7);
    ASSERT_EQ(metrics[0]->get_name(), "alpha_dir_x");
    ASSERT_EQ(metrics[2]->get_name(), "alpha_yaw_cos");
    ASSERT_EQ(metrics[4]->get_name(), "alpha_pitch");
    ASSERT_EQ(metrics[5]->get_name(), "alpha_fire");
    ASSERT_EQ(metrics[6]->get_name(), "alpha_zoom");
    for (const auto &metric: metrics) ASSERT_NEAR(metric->compute_metric(), 1e-3f, 1e-7f);
}

TEST_F(EntropyAlphasTest, EntropyUnderTargetRaisesAlphaAboveLowersIt) {
    const EntropyAlphas alphas(2, 2, 1e-2f, 1e-2f, std::vector(2, -1.f), 0.05f);

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

TEST_F(EntropyAlphasTest, MaskedDiscreteActionKeepsItsAlpha) {
    const EntropyAlphas alphas(1, 2, 1e-3f, 1e-2f, std::vector(1, -1.f), 0.05f);

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

TEST_F(EntropyAlphasTest, StepKeepsEveryAlphaAboveTheFloor) {
    const EntropyAlphas alphas(2, 2, 1e-2f, 1e-1f, std::vector(2, -1.f), 0.05f);

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

TEST_F(EntropyAlphasTest, EachContinuousActionFollowsItsOwnTarget) {
    const EntropyAlphas alphas(2, 1, 1e-2f, 1e-2f, {0.f, -1.f}, 0.05f);

    const auto continuous_entropy = torch::full({8, 2}, -0.5f);
    const auto discrete_entropy = torch::full({8, 1}, 0.2f);
    const auto mask = torch::ones({8, 1});

    for (int i = 0; i < 10; i++) {
        alphas.zero_grad();
        alphas.backward(continuous_entropy, discrete_entropy, mask, 1.f);
        alphas.step();
    }

    ASSERT_GT(alphas.continuous()[0].item<float>(), 1e-2f);
    ASSERT_LT(alphas.continuous()[1].item<float>(), 1e-2f);
}

TEST_F(EntropyAlphasTest, ContinuousTargetSizeMustMatchActions) {
    ASSERT_THROW(EntropyAlphas(5, 2, 1e-3f, 3e-4f, {0.f, 0.f, -1.f}, 0.05f), std::invalid_argument);
}

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

TEST_F(PidLagrangianAlphaParameterTest, SustainedOvershootParksAlphaAtZero) {
    const PidLagrangianAlphaParameters pid(2e-1f, 5e-3f, 1.f, 1e-3f, 1);

    const auto target = torch::full({8, 1}, 0.7f);

    for (int i = 0; i < 1000; i++) pid.update(torch::full({8, 1}, 5.f), target);

    ASSERT_FLOAT_EQ(pid.alpha().item<float>(), 0.f);
}

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
