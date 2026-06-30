//
// Created by samuel on 30/06/2026.
//

#include <networks_utils/init.h>

#include <arenai_train_tests/tests_networks_utils/tests_init.h>

TEST_F(InitWeightsTest, HiddenLinearWeightsBounded) {
    torch::nn::Linear linear(32, 16);
    init_hidden_weights(*linear);

    const auto max_abs = torch::max(torch::abs(linear->weight)).item<float>();

    // kaiming normal with fan_in=32 gives std ≈ 0.25, values should stay reasonable
    ASSERT_LT(max_abs, 10.0f);
    ASSERT_GT(max_abs, 0.0f);
}

TEST_F(InitWeightsTest, HiddenLinearBiasZero) {
    torch::nn::Linear linear(32, 16);
    init_hidden_weights(*linear);

    ASSERT_TRUE(torch::allclose(linear->bias, torch::zeros_like(linear->bias)));
}

TEST_F(InitWeightsTest, MuOutputWeightsSmall) {
    torch::nn::Linear linear(32, 4);
    init_mu_output_weights(*linear);

    const auto max_abs = torch::max(torch::abs(linear->weight)).item<float>();
    ASSERT_LE(max_abs, 1e-3f);
}

TEST_F(InitWeightsTest, MuOutputBiasZero) {
    torch::nn::Linear linear(32, 4);
    init_mu_output_weights(*linear);

    ASSERT_TRUE(torch::allclose(linear->bias, torch::zeros_like(linear->bias)));
}

TEST_F(InitWeightsTest, SigmaOutputBiasIsLog01) {
    torch::nn::Linear linear(32, 4);
    init_sigma_output_weights(*linear);

    const auto expected_bias = torch::full_like(linear->bias, std::log(0.1f));
    ASSERT_TRUE(torch::allclose(linear->bias, expected_bias));
}
