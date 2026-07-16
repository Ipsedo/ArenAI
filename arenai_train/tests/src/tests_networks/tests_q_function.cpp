//
// Created by samuel on 30/06/2026.
//

#include <networks/q_function.h>

#include <arenai_train_tests/tests_networks/tests_q_function.h>

using namespace arenai;
using namespace arenai::train;

TEST_P(QFunctionTestParam, TestQFunctionExpectation) {
    const auto
        [layers, cont_actions_nb, discrete_actions_nb, sensors_nb, sensors_hidden_size,
         actions_hidden_size, batch_size] = GetParam();

    constexpr int input_channels = 3;
    constexpr int width = 32;
    constexpr int height = 32;

    QFunction q_function(
        height, width, sensors_nb, cont_actions_nb, discrete_actions_nb, sensors_hidden_size,
        actions_hidden_size, layers, {{input_channels, 4}, {4, 8}}, {2, 4});

    const auto image = torch::randint(
        255, {batch_size, input_channels, height, width},
        torch::TensorOptions().dtype(torch::kUInt8));
    const auto sensors = torch::randn({batch_size, sensors_nb});

    const auto continuous_actions = torch::rand({batch_size, cont_actions_nb}) * 2.f - 1.f;
    const auto discrete_actions =
        torch::softmax(torch::randn({batch_size, discrete_actions_nb}), -1);

    const auto value =
        q_function.value_expectation(image, sensors, continuous_actions, discrete_actions);

    ASSERT_EQ(value.ndimension(), 2);
    ASSERT_EQ(value.size(0), batch_size);
    ASSERT_EQ(value.size(1), 1);
}

TEST_P(QFunctionTestParam, TestQFunctionOHE) {
    const auto
        [layers, cont_actions_nb, discrete_actions_nb, sensors_nb, sensors_hidden_size,
         actions_hidden_size, batch_size] = GetParam();

    constexpr int input_channels = 3;
    constexpr int width = 32;
    constexpr int height = 32;

    QFunction q_function(
        height, width, sensors_nb, cont_actions_nb, discrete_actions_nb, sensors_hidden_size,
        actions_hidden_size, layers, {{input_channels, 4}, {4, 8}}, {2, 4});

    const auto image = torch::randint(
        255, {batch_size, input_channels, height, width},
        torch::TensorOptions().dtype(torch::kUInt8));
    const auto sensors = torch::randn({batch_size, sensors_nb});

    const auto continuous_actions = torch::rand({batch_size, cont_actions_nb}) * 2.f - 1.f;
    const auto chosen_actions_index =
        torch::randint(discrete_actions_nb, {batch_size, 1}).to(torch::kLong);

    auto discrete_actions_ohe = torch::zeros({batch_size, discrete_actions_nb});
    discrete_actions_ohe = discrete_actions_ohe.scatter_(1, chosen_actions_index, 1.0);

    const auto value =
        q_function.value_ohe(image, sensors, continuous_actions, discrete_actions_ohe);

    ASSERT_EQ(value.ndimension(), 2);
    ASSERT_EQ(value.size(0), batch_size);
    ASSERT_EQ(value.size(1), 1);
}

INSTANTIATE_TEST_SUITE_P(
    TestQFunction, QFunctionTestParam,
    testing::Combine(
        testing::Values(HiddenLayers{16, 32}, HiddenLayers{2, 3}), testing::Values(1, 2, 3),
        testing::Values(2, 3, 4), testing::Values(1, 2, 3), testing::Values(2, 3, 4),
        testing::Values(2, 3, 4), testing::Values(1, 2, 3)));
