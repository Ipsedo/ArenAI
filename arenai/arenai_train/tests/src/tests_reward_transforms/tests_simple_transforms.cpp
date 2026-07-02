//
// Created by samuel on 30/06/2026.
//

#include <reward_transforms/add_combiner.h>
#include <reward_transforms/identity_transform.h>
#include <reward_transforms/scale_potential.h>

#include <arenai_train_tests/tests_reward_transforms/tests_simple_transforms.h>

using namespace arenai;
using namespace arenai::train;

// ========================================================================
// IdentityTransform
// ========================================================================

TEST_F(IdentityTransformTest, TransformReturnsInput) {
    IdentityTransform transform;

    const auto input = torch::tensor({1.0f, -2.0f, 3.5f});
    const auto result = transform.transform(input);

    ASSERT_TRUE(torch::equal(result, input));
}

TEST_F(IdentityTransformTest, OnAddIsNoOp) {
    IdentityTransform transform;

    ASSERT_NO_THROW(transform.on_add(torch::tensor(42.0f)));

    const auto input = torch::tensor({1.0f});
    const auto before = transform.transform(input);

    transform.on_add(torch::tensor(999.0f));

    const auto after = transform.transform(input);

    ASSERT_TRUE(torch::equal(before, after));
}

// ========================================================================
// ScalePotentialTransform
// ========================================================================

TEST_F(ScalePotentialTransformTest, ScalesCorrectly) {
    constexpr float scale = 2.5f;
    ScalePotentialTransform transform(scale);

    const auto input = torch::tensor({1.0f, -2.0f, 0.0f, 4.0f});
    const auto result = transform.transform(input);
    const auto expected = input * scale;

    ASSERT_TRUE(torch::allclose(result, expected));
}

TEST_F(ScalePotentialTransformTest, ZeroScale) {
    ScalePotentialTransform transform(0.0f);

    const auto input = torch::tensor({1.0f, -2.0f, 100.0f});
    const auto result = transform.transform(input);

    ASSERT_TRUE(torch::allclose(result, torch::zeros_like(input)));
}

TEST_F(ScalePotentialTransformTest, NegativeScale) {
    constexpr float scale = -1.0f;
    ScalePotentialTransform transform(scale);

    const auto input = torch::tensor({3.0f, -4.0f});
    const auto result = transform.transform(input);

    ASSERT_TRUE(torch::allclose(result, -input));
}

TEST_F(ScalePotentialTransformTest, OnAddIsNoOp) {
    ScalePotentialTransform transform(2.0f);

    const auto input = torch::tensor({5.0f});
    const auto before = transform.transform(input);

    transform.on_add(torch::tensor(999.0f));

    const auto after = transform.transform(input);

    ASSERT_TRUE(torch::equal(before, after));
}

// ========================================================================
// AddCombiner
// ========================================================================

TEST_F(AddCombinerTest, CombinesRewards) {
    AddCombiner combiner;

    const auto main_reward = torch::tensor({1.0f, 2.0f, 3.0f});
    const auto potential_reward = torch::tensor({0.5f, -1.0f, 0.0f});

    const auto result = combiner.to_reward({main_reward, potential_reward});
    const auto expected = main_reward + potential_reward;

    ASSERT_TRUE(torch::equal(result, expected));
}

TEST_F(AddCombinerTest, ZeroPotential) {
    AddCombiner combiner;

    const auto main_reward = torch::tensor({1.0f, 2.0f});
    const auto potential_reward = torch::zeros({2});

    const auto result = combiner.to_reward({main_reward, potential_reward});

    ASSERT_TRUE(torch::equal(result, main_reward));
}

TEST_F(AddCombinerTest, BatchDimensions) {
    AddCombiner combiner;

    const auto main_reward = torch::randn({32, 1});
    const auto potential_reward = torch::randn({32, 1});

    const auto result = combiner.to_reward({main_reward, potential_reward});

    ASSERT_EQ(result.size(0), 32);
    ASSERT_EQ(result.size(1), 1);
    ASSERT_TRUE(torch::allclose(result, main_reward + potential_reward));
}
