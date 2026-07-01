//
// Created by claude on 01/07/2026.
//

#include <reward_transforms/ema_potential.h>

#include <arenai_train_tests/tests_reward_transforms/tests_ema_potential_edge.h>

TEST_F(EmaPotentialEdgeTest, TransformBeforeAnyOnAdd) {
    EmaPotentialTransform ema(1.0f, 0.999f);

    const auto batch = torch::tensor({1.0f, 2.0f, 3.0f});
    const auto result = ema.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
        << "Transform before on_add should produce finite output (using initial mean=0, var=1)";
}

TEST_F(EmaPotentialEdgeTest, TransformAfterSingleOnAdd) {
    EmaPotentialTransform ema(1.0f, 0.999f);

    ema.on_add(torch::tensor({5.0f}));

    const auto batch = torch::tensor({5.0f, 5.0f});
    const auto result = ema.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
        << "Transform after single on_add should produce finite output";

    for (int i = 0; i < 2; i++) {
        ASSERT_NEAR(result[i].item<float>(), 0.0f, 1e-4f)
            << "Transforming the same value as the mean should yield ~0";
    }
}

TEST_F(EmaPotentialEdgeTest, TransformWithZeroRewards) {
    EmaPotentialTransform ema(2.0f, 0.99f);

    for (int i = 0; i < 100; i++) ema.on_add(torch::tensor({0.0f}));

    const auto batch = torch::tensor({0.0f, 0.0f, 0.0f});
    const auto result = ema.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
        << "All-zero rewards should still produce finite output";
}

TEST_F(EmaPotentialEdgeTest, TransformWithConstantRewards) {
    EmaPotentialTransform ema(1.0f, 0.99f);

    for (int i = 0; i < 100; i++) ema.on_add(torch::tensor({42.0f}));

    const auto batch = torch::tensor({42.0f, 42.0f});
    const auto result = ema.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>());

    for (int i = 0; i < 2; i++) {
        ASSERT_NEAR(result[i].item<float>(), 0.0f, 0.5f)
            << "Constant rewards should converge: mean≈42, so transform(42) ≈ 0";
    }
}

TEST_F(EmaPotentialEdgeTest, ScaleFactorApplied) {
    EmaPotentialTransform ema_s1(1.0f, 0.99f);
    EmaPotentialTransform ema_s2(2.0f, 0.99f);

    for (int i = 0; i < 50; i++) {
        const auto r = torch::tensor({static_cast<float>(i)});
        ema_s1.on_add(r);
        ema_s2.on_add(r);
    }

    const auto batch = torch::tensor({10.0f});
    const auto r1 = ema_s1.transform(batch);
    const auto r2 = ema_s2.transform(batch);

    ASSERT_NEAR(r2.item<float>(), 2.0f * r1.item<float>(), 1e-4f)
        << "Scale factor should multiply the output linearly";
}

TEST_F(EmaPotentialEdgeTest, VeryLargeRewardsDoNotOverflow) {
    EmaPotentialTransform ema(1.0f, 0.999f);

    for (int i = 0; i < 10; i++) ema.on_add(torch::tensor({1e10f}));

    const auto batch = torch::tensor({1e10f});
    const auto result = ema.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
        << "Very large rewards should not cause overflow";
}
