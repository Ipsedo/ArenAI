//
// Created by samuel on 30/06/2026.
//

#include <reward_transforms/running_norm_transform.h>

#include <arenai_train_tests/tests_reward_transforms/tests_running_norm.h>

using namespace arenai;
using namespace arenai::train;

// ========================================================================
// NormalizedRewardTransform
// ========================================================================

TEST_F(NormalizedRewardTransformTest, ConstantRewardKeepsSignAndStaysFinite) {
    NormalizedRewardTransform transform(100, 1.0f);

    constexpr float constant = 5.0f;
    for (int i = 0; i < 50; i++) transform.on_add(torch::tensor(constant));

    const auto result = transform.transform(torch::tensor({constant}));

    // no mean subtraction: a constant positive reward must stay positive
    ASSERT_TRUE(torch::isfinite(result).item<bool>());
    ASSERT_GT(result.item<float>(), 0.0f);
}

TEST_F(NormalizedRewardTransformTest, KnownMeanStd) {
    NormalizedRewardTransform transform(3, 1.0f);

    transform.on_add(torch::tensor(1.0f));
    transform.on_add(torch::tensor(2.0f));
    transform.on_add(torch::tensor(3.0f));

    // mean = 2.0, var = (1+0+1)/3 = 2/3, std = sqrt(2/3 + 1e-8)
    const auto expected_std = std::sqrt(2.0f / 3.0f + 1e-8f);

    const auto result = transform.transform(torch::tensor({5.0f}));
    const auto expected = 5.0f / expected_std;

    ASSERT_NEAR(result.item<float>(), expected, 1e-4f);
}

TEST_F(NormalizedRewardTransformTest, CircularEviction) {
    NormalizedRewardTransform transform(3, 1.0f);

    transform.on_add(torch::tensor(100.0f));
    transform.on_add(torch::tensor(200.0f));
    transform.on_add(torch::tensor(300.0f));

    // overwrite oldest (100.0)
    transform.on_add(torch::tensor(1.0f));
    // overwrite oldest (200.0)
    transform.on_add(torch::tensor(2.0f));
    // overwrite oldest (300.0)
    transform.on_add(torch::tensor(3.0f));

    // now buffer = [1, 2, 3], same as KnownMeanStd
    const auto expected_std = std::sqrt(2.0f / 3.0f + 1e-8f);
    const auto result = transform.transform(torch::tensor({5.0f}));
    const auto expected = 5.0f / expected_std;

    ASSERT_NEAR(result.item<float>(), expected, 1e-4f);
}

TEST_P(NormalizedRewardTransformParamTest, ScaleFactorApplied) {
    const auto [memory_size, reward_scale] = GetParam();

    NormalizedRewardTransform transform(memory_size, reward_scale);

    for (int i = 0; i < memory_size; i++) transform.on_add(torch::tensor(static_cast<float>(i)));

    const auto result_scaled = transform.transform(torch::tensor({0.0f}));

    NormalizedRewardTransform transform_unit(memory_size, 1.0f);
    for (int i = 0; i < memory_size; i++)
        transform_unit.on_add(torch::tensor(static_cast<float>(i)));

    const auto result_unit = transform_unit.transform(torch::tensor({0.0f}));

    ASSERT_NEAR(result_scaled.item<float>(), reward_scale * result_unit.item<float>(), 1e-5f);
}

TEST_F(NormalizedRewardTransformTest, SingleValueNoNaN) {
    NormalizedRewardTransform transform(100, 1.0f);

    transform.on_add(torch::tensor(42.0f));

    const auto result = transform.transform(torch::tensor({42.0f}));

    ASSERT_TRUE(torch::isfinite(result).item<bool>());
}

TEST_F(NormalizedRewardTransformTest, BatchTransform) {
    NormalizedRewardTransform transform(10, 1.0f);

    for (int i = 0; i < 10; i++) transform.on_add(torch::tensor(static_cast<float>(i)));

    const auto batch = torch::tensor({0.0f, 5.0f, 10.0f});
    const auto result = transform.transform(batch);

    ASSERT_EQ(result.size(0), 3);
    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    NormalizedReward, NormalizedRewardTransformParamTest,
    testing::Combine(testing::Values(10, 50, 100), testing::Values(0.5f, 1.0f, 2.0f)));

// ========================================================================
// NormalizedNonZeroTransform
// ========================================================================

TEST_F(NormalizedNonZeroTransformTest, AllZerosNoDivisionByZero) {
    NormalizedNonZeroTransform transform(10);

    for (int i = 0; i < 10; i++) transform.on_add(torch::tensor(0.0f));

    const auto result = transform.transform(torch::tensor({1.0f}));

    ASSERT_TRUE(torch::isfinite(result).item<bool>());
}

TEST_F(NormalizedNonZeroTransformTest, NonZeroRmsCorrect) {
    NormalizedNonZeroTransform transform(10);

    transform.on_add(torch::tensor(3.0f));
    transform.on_add(torch::tensor(4.0f));

    // rms = sqrt((9 + 16) / 2 + 1e-8) = sqrt(12.5 + 1e-8)
    const auto expected_rms = std::sqrt(12.5f + 1e-8f);
    const auto result = transform.transform(torch::tensor({5.0f}));

    ASSERT_NEAR(result.item<float>(), 5.0f / expected_rms, 1e-4f);
}

TEST_F(NormalizedNonZeroTransformTest, MixedZeroNonZero) {
    NormalizedNonZeroTransform transform(10);

    transform.on_add(torch::tensor(0.0f));
    transform.on_add(torch::tensor(4.0f));
    transform.on_add(torch::tensor(0.0f));
    transform.on_add(torch::tensor(0.0f));

    // only one non-zero: rms = sqrt(16 / 1 + 1e-8) = sqrt(16 + 1e-8)
    const auto expected_rms = std::sqrt(16.0f + 1e-8f);
    const auto result = transform.transform(torch::tensor({8.0f}));

    ASSERT_NEAR(result.item<float>(), 8.0f / expected_rms, 1e-4f);
}

TEST_F(NormalizedNonZeroTransformTest, CircularEvictionNonZero) {
    NormalizedNonZeroTransform transform(3);

    transform.on_add(torch::tensor(5.0f));
    transform.on_add(torch::tensor(0.0f));
    transform.on_add(torch::tensor(0.0f));

    // overwrite the 5.0 (non-zero evicted)
    transform.on_add(torch::tensor(0.0f));

    // now all zeros in buffer
    const auto result = transform.transform(torch::tensor({1.0f}));

    // rms = sqrt(0 / max(0,1) + 1e-8) ≈ sqrt(1e-8)
    ASSERT_TRUE(torch::isfinite(result).item<bool>());
    ASSERT_GT(std::abs(result.item<float>()), 0.0f);
}

TEST_F(NormalizedNonZeroTransformTest, CircularEvictionZeroReplacedByNonZero) {
    NormalizedNonZeroTransform transform(3);

    transform.on_add(torch::tensor(0.0f));
    transform.on_add(torch::tensor(0.0f));
    transform.on_add(torch::tensor(0.0f));

    // overwrite first zero with non-zero
    transform.on_add(torch::tensor(6.0f));

    // non_zero_nb = 1, sum_sq = 36
    const auto expected_rms = std::sqrt(36.0f + 1e-8f);
    const auto result = transform.transform(torch::tensor({12.0f}));

    ASSERT_NEAR(result.item<float>(), 12.0f / expected_rms, 1e-4f);
}

TEST_P(NormalizedNonZeroTransformParamTest, BatchTransform) {
    const auto memory_size = GetParam();

    NormalizedNonZeroTransform transform(memory_size);

    for (int i = 0; i < memory_size; i++) transform.on_add(torch::tensor(static_cast<float>(i)));

    const auto batch = torch::tensor({1.0f, 2.0f, 3.0f});
    const auto result = transform.transform(batch);

    ASSERT_EQ(result.size(0), 3);
    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    NormalizedNonZero, NormalizedNonZeroTransformParamTest, testing::Values(5, 10, 50));
