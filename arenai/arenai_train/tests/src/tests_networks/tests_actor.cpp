//
// Created by samuel on 30/06/2026.
//

#include <networks/actor.h>

#include <arenai_train_tests/tests_networks/tests_actor.h>

TEST_P(ActorTestParam, TestActorAct) {
    const auto
        [layers, cont_actions_nb, discrete_actions_nb, sensors_nb, sensors_hidden_size,
         batch_size] = GetParam();

    constexpr int input_channels = 3;
    constexpr int width = 32;
    constexpr int height = 32;

    Actor actor(
        height, width, sensors_nb, cont_actions_nb, discrete_actions_nb, sensors_hidden_size,
        layers, {{input_channels, 4}, {4, 8}}, {2, 4});

    const auto image = torch::randint(
        255, {batch_size, input_channels, height, width},
        torch::TensorOptions().dtype(torch::kUInt8));
    const auto sensors = torch::randn({batch_size, sensors_nb});

    const auto [mu, sigma, discrete] = actor.act(image, sensors);

    ASSERT_EQ(mu.ndimension(), 2);
    ASSERT_EQ(mu.size(0), batch_size);
    ASSERT_EQ(mu.size(1), cont_actions_nb);
    ASSERT_TRUE(
        torch::all(torch::logical_and(torch::ge(mu, -1.0), torch::le(mu, 1.0))).item<bool>());

    ASSERT_EQ(sigma.ndimension(), 2);
    ASSERT_EQ(sigma.size(0), batch_size);
    ASSERT_EQ(sigma.size(1), cont_actions_nb);
    ASSERT_TRUE(
        torch::all(torch::logical_and(torch::gt(sigma, 0.0), torch::le(sigma, 1.0))).item<bool>());

    ASSERT_EQ(discrete.ndimension(), 2);
    ASSERT_EQ(discrete.size(0), batch_size);
    ASSERT_EQ(discrete.size(1), discrete_actions_nb);
    ASSERT_TRUE(torch::all(torch::abs(torch::sum(discrete, -1) - 1.0) < 1e-6).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    TestActor, ActorTestParam,
    testing::Combine(
        testing::Values(HiddenLayers{16, 32}, HiddenLayers{2, 3}), testing::Values(1, 2, 3),
        testing::Values(2, 3, 4), testing::Values(1, 2, 3), testing::Values(2, 3, 4),
        testing::Values(1, 2, 3)));
