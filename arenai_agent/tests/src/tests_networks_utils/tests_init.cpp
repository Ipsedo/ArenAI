//
// Created by samuel on 30/06/2026.
//

#include <networks_utils/init.h>

#include <arenai_agent_tests/tests_networks_utils/tests_init.h>

using namespace arenai;
using namespace arenai::agent;

namespace {
    // orthogonal_ rows (flattened) satisfy W @ W^T = gain^2 * I when out_features <= in_features
    void assert_orthogonal(const torch::Tensor &weight, const float gain) {
        const auto w = weight.reshape({weight.size(0), -1});
        const auto gram = torch::mm(w, w.t());
        const auto expected = gain * gain * torch::eye(w.size(0));
        ASSERT_TRUE(torch::allclose(gram, expected, 1e-4, 1e-6));
    }
}// namespace

TEST_F(InitWeightsTest, HiddenLinearWeightsOrthogonal) {
    torch::nn::Linear linear(32, 16);
    init_hidden_weights(*linear);

    assert_orthogonal(linear->weight, std::sqrt(2.f));
}

TEST_F(InitWeightsTest, HiddenConvWeightsOrthogonal) {
    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(4, 8, 3));
    init_hidden_weights(*conv);

    assert_orthogonal(conv->weight, std::sqrt(2.f));
}

TEST_F(InitWeightsTest, HiddenLinearBiasZero) {
    torch::nn::Linear linear(32, 16);
    init_hidden_weights(*linear);

    ASSERT_TRUE(torch::allclose(linear->bias, torch::zeros_like(linear->bias)));
}

TEST_F(InitWeightsTest, MuOutputWeightsOrthogonal) {
    torch::nn::Linear linear(32, 4);
    init_mu_output_weights(*linear);

    assert_orthogonal(linear->weight, 0.01f);
}

TEST_F(InitWeightsTest, MuOutputBiasZero) {
    torch::nn::Linear linear(32, 4);
    init_mu_output_weights(*linear);

    ASSERT_TRUE(torch::allclose(linear->bias, torch::zeros_like(linear->bias)));
}

TEST_F(InitWeightsTest, SigmaOutputWeightsOrthogonal) {
    torch::nn::Linear linear(32, 4);
    init_sigma_output_weights(*linear);

    assert_orthogonal(linear->weight, 0.01f);
}

TEST_F(InitWeightsTest, SigmaOutputBiasIsLogOne) {
    torch::nn::Linear linear(32, 4);
    init_sigma_output_weights(*linear);

    const auto expected_bias = torch::full_like(linear->bias, std::log(1.f));
    ASSERT_TRUE(torch::allclose(linear->bias, expected_bias));
}

TEST_F(InitWeightsTest, DiscreteOutputWeightsOrthogonal) {
    torch::nn::Linear linear(32, 6);
    init_discrete_output_weights(*linear);

    assert_orthogonal(linear->weight, 0.01f);
}

TEST_F(InitWeightsTest, ValueOutputWeightsOrthogonal) {
    torch::nn::Linear linear(32, 1);
    init_value_output_weights(*linear);

    assert_orthogonal(linear->weight, 1.f);
}
