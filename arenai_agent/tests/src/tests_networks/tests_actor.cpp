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

    const auto [mode, concentration, discrete] = actor.act(image, sensors);

    ASSERT_EQ(mode.ndimension(), 2);
    ASSERT_EQ(mode.size(0), batch_size);
    ASSERT_EQ(mode.size(1), cont_actions_nb);
    ASSERT_TRUE(
        torch::all(torch::logical_and(torch::ge(mode, 0.0), torch::le(mode, 1.0))).item<bool>());

    ASSERT_EQ(concentration.ndimension(), 2);
    ASSERT_EQ(concentration.size(0), batch_size);
    ASSERT_EQ(concentration.size(1), cont_actions_nb);
    ASSERT_TRUE(torch::all(torch::logical_and(
                               torch::ge(concentration, CONCENTRATION_MIN),
                               torch::le(concentration, CONCENTRATION_MAX)))
                    .item<bool>());

    ASSERT_EQ(discrete.ndimension(), 2);
    ASSERT_EQ(discrete.size(0), batch_size);
    ASSERT_EQ(discrete.size(1), discrete_actions_nb);
    // independent Bernoulli probabilities: each in (0, 1), no sum constraint
    ASSERT_TRUE(torch::all(torch::logical_and(torch::gt(discrete, 0.0), torch::lt(discrete, 1.0)))
                    .item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    TestActor, ActorTestParam,
    testing::Combine(
        testing::Values(HiddenLayers{16, 32}, HiddenLayers{2, 3}), testing::Values(1, 2, 3),
        testing::Values(2, 3, 4), testing::Values(1, 2, 3), testing::Values(2, 3, 4),
        testing::Values(1, 2, 3)));
