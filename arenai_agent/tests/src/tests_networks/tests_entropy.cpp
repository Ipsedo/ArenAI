//
// Created by samuel on 30/06/2026.
//

#include <networks/entropy.h>

#include <arenai_agent_tests/tests_networks/tests_entropy.h>

using namespace arenai;
using namespace arenai::agent;

TEST_F(AlphaParameterTest, AlphaAlwaysPositive) {
    for (const float init: {0.01f, 0.1f, 1.0f, 10.0f}) {
        AlphaParameters param(init, 1);
        ASSERT_GT(param.alpha().item<float>(), 0.0f)
            << "alpha should be positive for initial_alpha=" << init;
    }
}

TEST_F(AlphaParameterTest, InitialValueMatchesInput) {
    constexpr float initial = 0.2f;
    AlphaParameters param(initial, 1);

    ASSERT_NEAR(param.alpha().item<float>(), initial, 1e-6f);
}

TEST_F(AlphaParameterTest, LogAlphaRequiresGrad) {
    AlphaParameters param(1.0f, 1);

    ASSERT_TRUE(param.log_alpha().requires_grad());
}

TEST_F(AlphaParameterTest, LogAlphaConsistentWithAlpha) {
    AlphaParameters param(0.5f, 1);

    const auto log_a = param.log_alpha().item<float>();
    const auto a = param.alpha().item<float>();

    ASSERT_NEAR(std::exp(log_a), a, 1e-6f);
}
