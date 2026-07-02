//
// Created by samuel on 30/06/2026.
//

#include <networks/misc.h>

#include <arenai_train_tests/tests_networks/tests_misc.h>

using namespace arenai;
using namespace arenai::train;

// ========================================================================
// Clamp
// ========================================================================

TEST_P(ClampModuleParamTest, ClampsWithinBounds) {
    const auto [lower, upper, shape] = GetParam();

    Clamp clamp(lower, upper);

    const auto input = torch::randn(shape) * 10.0f;
    const auto result = clamp.forward(input);

    ASSERT_EQ(result.sizes(), input.sizes());
    ASSERT_TRUE(torch::all(torch::logical_and(torch::ge(result, lower), torch::le(result, upper)))
                    .item<bool>());
}

TEST_P(ClampModuleParamTest, ValuesInsideBoundsUnchanged) {
    const auto [lower, upper, shape] = GetParam();

    Clamp clamp(lower, upper);

    const auto mid = (lower + upper) / 2.0f;
    const auto range = (upper - lower) / 2.0f;
    const auto input = torch::rand(shape) * range * 0.5f + mid - range * 0.25f;
    const auto result = clamp.forward(input);

    ASSERT_TRUE(torch::allclose(result, input));
}

INSTANTIATE_TEST_SUITE_P(
    TestClamp, ClampModuleParamTest,
    testing::Combine(
        testing::Values(-1.0f, -0.5f, 0.0f), testing::Values(0.5f, 1.0f, 2.0f),
        testing::Values(Shape{5}, Shape{2, 3}, Shape{4, 8, 2})));

// ========================================================================
// Exp
// ========================================================================

TEST_P(ExpModuleParamTest, OutputAlwaysPositive) {
    const auto shape = GetParam();

    Exp exp_module;

    const auto input = torch::randn(shape) * 5.0f;
    const auto result = exp_module.forward(input);

    ASSERT_EQ(result.sizes(), input.sizes());
    ASSERT_TRUE(torch::all(torch::gt(result, 0.0f)).item<bool>());
}

TEST_P(ExpModuleParamTest, MatchesTorchExp) {
    const auto shape = GetParam();

    Exp exp_module;

    const auto input = torch::randn(shape);
    const auto result = exp_module.forward(input);
    const auto expected = torch::exp(input);

    ASSERT_TRUE(torch::allclose(result, expected));
}

INSTANTIATE_TEST_SUITE_P(
    TestExp, ExpModuleParamTest, testing::Values(Shape{1}, Shape{5}, Shape{2, 3}, Shape{4, 8, 2}));
