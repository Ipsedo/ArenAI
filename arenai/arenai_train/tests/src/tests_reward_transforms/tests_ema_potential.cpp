//
// Created by samuel on 30/06/2026.
//

#include <reward_transforms/ema_potential.h>

#include <arenai_train_tests/tests_reward_transforms/tests_ema_potential.h>

using namespace arenai;
using namespace arenai::train;

TEST_F(EmaPotentialTransformTest, FirstSampleInitializesMean) {
    EmaPotentialTransform transform(1.0f, 0.999f);

    transform.on_add(torch::tensor(7.0f));

    // after first sample: mean = 7.0, var = 1.0, std = sqrt(1 + 1e-8)
    // transform(7.0) = scale * (7 - 7) / std = 0
    const auto result = transform.transform(torch::tensor({7.0f}));

    ASSERT_NEAR(result.item<float>(), 0.0f, 1e-5f);
}

TEST_F(EmaPotentialTransformTest, ConvergenceToConstant) {
    EmaPotentialTransform transform(1.0f, 0.99f);

    constexpr float constant = 3.0f;
    for (int i = 0; i < 1000; i++) transform.on_add(torch::tensor(constant));

    const auto result = transform.transform(torch::tensor({constant}));

    ASSERT_NEAR(result.item<float>(), 0.0f, 1e-2f);
}

TEST_F(EmaPotentialTransformTest, ScaleApplied) {
    constexpr float scale = 2.5f;
    EmaPotentialTransform transform_scaled(scale, 0.9f);
    EmaPotentialTransform transform_unit(1.0f, 0.9f);

    for (int i = 0; i < 20; i++) {
        const auto val = torch::tensor(static_cast<float>(i));
        transform_scaled.on_add(val);
        transform_unit.on_add(val);
    }

    const auto input = torch::tensor({10.0f});
    const auto result_scaled = transform_scaled.transform(input);
    const auto result_unit = transform_unit.transform(input);

    ASSERT_NEAR(result_scaled.item<float>(), scale * result_unit.item<float>(), 1e-5f);
}

TEST_F(EmaPotentialTransformTest, TransformFiniteAfterFewSamples) {
    EmaPotentialTransform transform(1.0f, 0.999f);

    transform.on_add(torch::tensor(1.0f));
    transform.on_add(torch::tensor(-1.0f));
    transform.on_add(torch::tensor(0.5f));

    const auto result = transform.transform(torch::tensor({0.0f, 1.0f, -1.0f}));

    ASSERT_EQ(result.size(0), 3);
    ASSERT_TRUE(torch::all(torch::isfinite(result)).item<bool>());
}

TEST_F(EmaPotentialTransformTest, EmaDecayAffectsConvergence) {
    EmaPotentialTransform fast_decay(1.0f, 0.5f);
    EmaPotentialTransform slow_decay(1.0f, 0.999f);

    // feed a step change: 10 zeros then 10 ones
    for (int i = 0; i < 10; i++) {
        fast_decay.on_add(torch::tensor(0.0f));
        slow_decay.on_add(torch::tensor(0.0f));
    }
    for (int i = 0; i < 10; i++) {
        fast_decay.on_add(torch::tensor(1.0f));
        slow_decay.on_add(torch::tensor(1.0f));
    }

    // fast decay should have adapted more: transform(1.0) closer to 0
    const auto fast_result = fast_decay.transform(torch::tensor({1.0f}));
    const auto slow_result = slow_decay.transform(torch::tensor({1.0f}));

    ASSERT_LT(std::abs(fast_result.item<float>()), std::abs(slow_result.item<float>()));
}

TEST_F(EmaPotentialTransformTest, LargeValueNoOverflow) {
    EmaPotentialTransform transform(1.0f, 0.99f);

    transform.on_add(torch::tensor(1e6f));
    transform.on_add(torch::tensor(-1e6f));
    transform.on_add(torch::tensor(1e6f));

    const auto result = transform.transform(torch::tensor({0.0f}));

    ASSERT_TRUE(torch::isfinite(result).item<bool>());
}
