//
// Created by samuel on 30/06/2026.
//

#include <networks/entropy.h>

#include <arenai_train_tests/tests_networks/tests_entropy.h>

using namespace arenai;
using namespace arenai::train;

TEST_F(AlphaParameterTest, AlphaAlwaysPositive) {
    for (const float init: {0.01f, 0.1f, 1.0f, 10.0f}) {
        AlphaParameter param(init);
        ASSERT_GT(param.alpha().item<float>(), 0.0f)
            << "alpha should be positive for initial_alpha=" << init;
    }
}

TEST_F(AlphaParameterTest, InitialValueMatchesInput) {
    constexpr float initial = 0.2f;
    AlphaParameter param(initial);

    ASSERT_NEAR(param.alpha().item<float>(), initial, 1e-6f);
}

TEST_F(AlphaParameterTest, LogAlphaRequiresGrad) {
    AlphaParameter param(1.0f);

    ASSERT_TRUE(param.log_alpha().requires_grad());
}

TEST_F(AlphaParameterTest, LogAlphaConsistentWithAlpha) {
    AlphaParameter param(0.5f);

    const auto log_a = param.log_alpha().item<float>();
    const auto a = param.alpha().item<float>();

    ASSERT_NEAR(std::exp(log_a), a, 1e-6f);
}
