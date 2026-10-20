//
// Created by samuel on 30/06/2026.
//

#include <networks/actor.h>
#include <networks/constants.h>

#include <arenai_agent_tests/tests_networks/tests_actor.h>

using namespace arenai;
using namespace arenai::agent;

TEST_P(ActorTestParam, TestActorAct) {
    const auto
        [layers, cont_actions_nb, discrete_actions_nb, sensors_nb, sensors_hidden_size,
         batch_size] = GetParam();

    constexpr int input_channels = 3;
    constexpr int width = 32;
    constexpr int height = 32;

    Actor actor(
        height, width, sensors_nb, cont_actions_nb, discrete_actions_nb, sensors_hidden_size,
        layers, {{input_channels, 4}, {4, 8}}, {2, 4}, 0.1f,
        std::vector(discrete_actions_nb, 0.2f));

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
        torch::all(torch::logical_and(torch::ge(sigma, SIGMA_MIN), torch::le(sigma, SIGMA_MAX)))
            .item<bool>());

    ASSERT_EQ(discrete.ndimension(), 2);
    ASSERT_EQ(discrete.size(0), batch_size);
    ASSERT_EQ(discrete.size(1), discrete_actions_nb);
    // independent Bernoulli probabilities: each in (0, 1), no sum constraint
    ASSERT_TRUE(torch::all(torch::logical_and(torch::gt(discrete, 0.0), torch::lt(discrete, 1.0)))
                    .item<bool>());
}

TEST(ActorTest, SigmaIsStateIndependentAndStartsAtInitialSigma) {
    constexpr int height = 32, width = 32, nb_sensors = 5, nb_continuous = 4, batch_size = 3;
    constexpr float initial_sigma = 0.5f;

    Actor actor(
        height, width, nb_sensors, nb_continuous, 2, 8, {16}, {{3, 4}, {4, 8}}, {2, 4},
        initial_sigma, std::vector(2, 0.2f));

    const auto image = torch::randint(
        255, {batch_size, 3, height, width}, torch::TensorOptions().dtype(torch::kUInt8));
    const auto sensors = torch::randn({batch_size, nb_sensors});

    const auto sigma = actor.act(image, sensors).sigma;

    ASSERT_TRUE(torch::allclose(sigma, torch::full_like(sigma, initial_sigma), 1e-4, 1e-5));
    ASSERT_TRUE(torch::equal(sigma[0], sigma[1]));
}

INSTANTIATE_TEST_SUITE_P(
    TestActor, ActorTestParam,
    testing::Combine(
        testing::Values(HiddenLayers{16, 32}, HiddenLayers{2, 3}), testing::Values(1, 2, 3),
        testing::Values(2, 3, 4), testing::Values(1, 2, 3), testing::Values(2, 3, 4),
        testing::Values(1, 2, 3)));
