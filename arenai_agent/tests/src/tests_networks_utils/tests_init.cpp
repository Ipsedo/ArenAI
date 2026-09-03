//
// Created by samuel on 30/06/2026.
//

#include <distributions/beta_law.h>
#include <networks_utils/init.h>

#include <arenai_agent_tests/tests_networks_utils/tests_init.h>
#include <arenai_model/constants.h>

#include "./networks/constants.h"
#include "./networks/misc.h"

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

TEST_F(InitWeightsTest, KappaOutputWeightsOrthogonal) {
    torch::nn::Linear linear(32, 4);
    init_kappa_output_weights(*linear, 0.05f);

    assert_orthogonal(linear->weight, 0.01f);
}

TEST_F(InitWeightsTest, KappaOutputIsEqualToWantedOne) {
    // same output activation as the actor's kappa head
    torch::nn::Sequential sequential(
        torch::nn::Linear(32, 4), std::make_shared<RangeSigmoidOutput>(MIN_KAPPA_SIGMOID, 1.f));

    constexpr float wanted_kappa = 0.5f;

    sequential->apply([](torch::nn::Module &m) { init_kappa_output_weights(m, wanted_kappa); });

    const auto x = torch::zeros({3, 32});
    const auto out = sequential->forward(x);

    ASSERT_NEAR(out.mean().item<float>(), wanted_kappa, 1e-3);
}

TEST_F(InitWeightsTest, DiscreteOutputWeightsOrthogonal) {
    torch::nn::Linear linear(32, 6);
    init_discrete_output_weights(*linear, 0.f);

    assert_orthogonal(linear->weight, 0.01f);
}

TEST_F(InitWeightsTest, DiscreteOutputFireProbaIsEqualToWantedOne) {
    constexpr float wanted_fire_proba = 0.2f;

    torch::nn::Sequential seq(
        torch::nn::Linear(32, model::ENEMY_NB_DISCRETE_ACTION), torch::nn::Softmax(-1));
    seq->apply([](torch::nn::Module &m) { init_discrete_output_weights(m, wanted_fire_proba); });

    torch::Tensor x = torch::randn({1, 32});
    const auto out = seq->forward(x);

    ASSERT_NEAR(out[0][0].item<float>(), wanted_fire_proba, 1e-2f);
    ASSERT_NEAR(out[0][1].item<float>(), 1.f - wanted_fire_proba, 1e-2f);
}

TEST_F(InitWeightsTest, ValueOutputWeightsOrthogonal) {
    torch::nn::Linear linear(32, 1);
    init_value_output_weights(*linear);

    assert_orthogonal(linear->weight, 1.f);
}
