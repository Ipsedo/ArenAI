//
// Created by claude on 01/07/2026.
//

#include <reward_transforms/running_norm.h>

#include <arenai_train_tests/tests_reward_transforms/tests_running_norm_edge.h>

TEST_F(RunningNormEdgeTest, TransformBeforeAnyOnAdd) {
    NormalizedRewardTransform norm(10, 1.0f);

    const auto batch = torch::tensor({1.0f, 2.0f, 3.0f});
    const auto result = norm.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
        << "Transform before on_add should produce finite results (even if degenerate)";
}

TEST_F(RunningNormEdgeTest, TransformAfterSingleOnAdd) {
    NormalizedRewardTransform norm(10, 1.0f);

    norm.on_add(torch::tensor({5.0f}));

    const auto batch = torch::tensor({5.0f});
    const auto result = norm.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
        << "Single sample should not cause div-by-zero";
}

TEST_F(RunningNormEdgeTest, TransformWithConstantRewards) {
    NormalizedRewardTransform norm(10, 1.0f);

    for (int i = 0; i < 10; i++) norm.on_add(torch::tensor({7.0f}));

    const auto batch = torch::tensor({7.0f});
    const auto result = norm.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
        << "Zero variance from constant rewards should not produce NaN/Inf";
}

TEST_F(RunningNormEdgeTest, NormalizationProducesCorrectDirection) {
    NormalizedRewardTransform norm(100, 1.0f);

    for (int i = 0; i < 20; i++) norm.on_add(torch::tensor({static_cast<float>(i)}));

    const auto high = norm.transform(torch::tensor({100.0f}));
    const auto low = norm.transform(torch::tensor({-100.0f}));

    ASSERT_GT(high.item<float>(), low.item<float>())
        << "Higher raw reward should yield higher normalized reward";
}

TEST_F(RunningNormEdgeTest, ScaleFactorApplied) {
    NormalizedRewardTransform norm1(100, 1.0f);
    NormalizedRewardTransform norm3(100, 3.0f);

    for (int i = 0; i < 50; i++) {
        const auto r = torch::tensor({static_cast<float>(i % 10)});
        norm1.on_add(r);
        norm3.on_add(r);
    }

    const auto batch = torch::tensor({5.0f});
    const auto r1 = norm1.transform(batch);
    const auto r3 = norm3.transform(batch);

    ASSERT_NEAR(r3.item<float>(), 3.0f * r1.item<float>(), 1e-3f)
        << "reward_scale should multiply the normalized output";
}

TEST_F(RunningNormEdgeTest, CircularBufferWraparound) {
    constexpr int mem_size = 5;
    NormalizedRewardTransform norm(mem_size, 1.0f);

    for (int i = 0; i < 3 * mem_size; i++) {
        norm.on_add(torch::tensor({static_cast<float>(i)}));

        const auto batch = torch::tensor({static_cast<float>(i)});
        const auto result = norm.transform(batch);

        ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
            << "Should remain finite through circular buffer wraparound at step " << i;
    }
}

TEST_F(RunningNormEdgeTest, FloatingPointDriftWithManyUpdates) {
    NormalizedRewardTransform norm(100, 1.0f);

    for (int i = 0; i < 10000; i++) {
        const float r = static_cast<float>(i % 7) * 0.1f;
        norm.on_add(torch::tensor({r}));
    }

    const auto batch = torch::tensor({0.3f});
    const auto result = norm.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
        << "Should remain finite after 10k updates (potential floating-point drift)";
}

TEST_F(RunningNormEdgeTest, NormalizedNonZeroWithAllZeros) {
    NormalizedNonZeroTransform norm(10);

    for (int i = 0; i < 10; i++) norm.on_add(torch::tensor({0.0f}));

    const auto batch = torch::tensor({0.0f, 1.0f, -1.0f});
    const auto result = norm.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
        << "All-zero rewards should not cause NaN in NormalizedNonZero (non_zero_nb_=0)";
}

TEST_F(RunningNormEdgeTest, NormalizedNonZeroBeforeOnAdd) {
    NormalizedNonZeroTransform norm(10);

    const auto batch = torch::tensor({1.0f, 2.0f});
    const auto result = norm.transform(batch);

    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>())
        << "NormalizedNonZero transform before any on_add should be finite";
}
