#include <networks/misc.h>

#include <arenai_agent_tests/tests_networks/tests_misc.h>

using namespace arenai;
using namespace arenai::agent;

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

TEST_P(SoftClampModuleParamTest, StaysStrictlyWithinBounds) {
    const auto [lower, upper, shape] = GetParam();

    SoftClamp soft_clamp(lower, upper);

    // moderate inputs: a sigmoid of ±30 rounds to the bound itself in float32
    const auto input = torch::randn(shape) * 3.0f;
    const auto result = soft_clamp.forward(input);

    ASSERT_EQ(result.sizes(), input.sizes());
    ASSERT_TRUE(torch::all(torch::logical_and(torch::gt(result, lower), torch::lt(result, upper)))
                    .item<bool>());
}

TEST_P(SoftClampModuleParamTest, ZeroMapsToMidpoint) {
    const auto [lower, upper, shape] = GetParam();

    SoftClamp soft_clamp(lower, upper);

    const auto result = soft_clamp.forward(torch::zeros(shape));

    ASSERT_TRUE(torch::allclose(result, torch::full(shape, (lower + upper) / 2.0f)));
}

TEST_P(SoftClampModuleParamTest, GradientNeverVanishes) {
    const auto [lower, upper, shape] = GetParam();

    SoftClamp soft_clamp(lower, upper);

    // far outside the bounds, where a hard clamp has a zero gradient
    const auto input = (torch::randn(shape) * 4.0f).set_requires_grad(true);
    soft_clamp.forward(input).sum().backward();

    ASSERT_TRUE(torch::all(torch::gt(input.grad(), 0.0f)).item<bool>());
}

TEST_P(SoftClampModuleParamTest, InverseRoundTrips) {
    const auto [lower, upper, shape] = GetParam();

    SoftClamp soft_clamp(lower, upper);

    for (const float wanted:
         {lower + 0.1f * (upper - lower), (lower + upper) / 2.0f, upper - 0.1f * (upper - lower)}) {
        const auto pre_activation = SoftClamp::inverse(wanted, lower, upper);
        const auto result = soft_clamp.forward(torch::full(shape, pre_activation));

        ASSERT_TRUE(torch::allclose(result, torch::full(shape, wanted), 1e-4, 1e-5));
    }
}

INSTANTIATE_TEST_SUITE_P(
    TestSoftClamp, SoftClampModuleParamTest,
    testing::Combine(
        testing::Values(-5.0f, -1.0f), testing::Values(-0.3567f, 0.5f, 2.0f),
        testing::Values(Shape{5}, Shape{2, 3}, Shape{4, 8, 2})));

TEST_P(ExpModuleParamTest, OutputAlwaysPositive) {
    const auto &shape = GetParam();

    Exp exp_module;

    const auto input = torch::randn(shape) * 5.0f;
    const auto result = exp_module.forward(input);

    ASSERT_EQ(result.sizes(), input.sizes());
    ASSERT_TRUE(torch::all(torch::gt(result, 0.0f)).item<bool>());
}

TEST_P(ExpModuleParamTest, MatchesTorchExp) {
    const auto &shape = GetParam();

    Exp exp_module;

    const auto input = torch::randn(shape);
    const auto result = exp_module.forward(input);
    const auto expected = torch::exp(input);

    ASSERT_TRUE(torch::allclose(result, expected));
}

INSTANTIATE_TEST_SUITE_P(
    TestExp, ExpModuleParamTest, testing::Values(Shape{1}, Shape{5}, Shape{2, 3}, Shape{4, 8, 2}));
