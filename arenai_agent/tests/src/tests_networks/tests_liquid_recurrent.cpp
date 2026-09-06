//
// Created by samuel on 06/09/2026.
//

#include <networks/recurrent/liquid_recurrent.h>

#include <arenai_agent_tests/tests_networks/tests_liquid_recurrent.h>

namespace {
    constexpr float DELTA_T = 1.f;

    torch::Tensor tanh_activation(const torch::Tensor &t) { return torch::tanh(t); }

    void assert_all_parameters_receive_gradient(const torch::nn::Module &module) {
        for (const auto &p: module.named_parameters()) {
            ASSERT_TRUE(p.value().grad().defined())
                << "Parameter \"" << p.key() << "\" has no gradient";
            ASSERT_TRUE(torch::all(torch::isfinite(p.value().grad())).item<bool>())
                << "Parameter \"" << p.key() << "\" has non-finite gradients";
            ASSERT_GT(p.value().grad().abs().sum().item<float>(), 0.f)
                << "Parameter \"" << p.key() << "\" has an all-zero gradient";
        }
    }
}// namespace

/*
 * Cell model
 */

TEST_P(CellModelTestParam, OutputShape) {
    const auto [neuron_number, input_size, batch_size] = GetParam();

    CellModel cell_model(neuron_number, input_size, tanh_activation);

    const auto x_t = torch::randn({batch_size, neuron_number});
    const auto input_t = torch::randn({batch_size, input_size});

    const auto output = cell_model.forward(x_t, input_t);

    ASSERT_EQ(output.ndimension(), 2);
    ASSERT_EQ(output.size(0), batch_size);
    ASSERT_EQ(output.size(1), neuron_number);
}

TEST_P(CellModelTestParam, GradientFlows) {
    const auto [neuron_number, input_size, batch_size] = GetParam();

    CellModel cell_model(neuron_number, input_size, tanh_activation);

    const auto x_t = torch::randn({batch_size, neuron_number}, torch::requires_grad());
    const auto input_t = torch::randn({batch_size, input_size}, torch::requires_grad());

    cell_model.forward(x_t, input_t).sum().backward();

    assert_all_parameters_receive_gradient(cell_model);

    ASSERT_TRUE(x_t.grad().defined());
    ASSERT_GT(x_t.grad().abs().sum().item<float>(), 0.f);
    ASSERT_TRUE(input_t.grad().defined());
    ASSERT_GT(input_t.grad().abs().sum().item<float>(), 0.f);
}

INSTANTIATE_TEST_SUITE_P(
    TestCellModel, CellModelTestParam,
    testing::Combine(testing::Values(2, 8, 16), testing::Values(1, 3, 12), testing::Values(1, 4)));

/*
 * Liquid cell
 */

TEST_P(LiquidCellTestParam, OutputShape) {
    const auto [neuron_number, input_size, unfolding_steps, batch_size] = GetParam();

    LiquidCell liquid_cell(neuron_number, input_size, unfolding_steps, tanh_activation, DELTA_T);

    const auto x_t = torch::randn({batch_size, neuron_number});
    const auto input_t = torch::randn({batch_size, input_size});

    const auto output = liquid_cell.forward(x_t, input_t);

    ASSERT_EQ(output.ndimension(), 2);
    ASSERT_EQ(output.size(0), batch_size);
    ASSERT_EQ(output.size(1), neuron_number);
}

TEST_P(LiquidCellTestParam, GradientFlows) {
    const auto [neuron_number, input_size, unfolding_steps, batch_size] = GetParam();

    LiquidCell liquid_cell(neuron_number, input_size, unfolding_steps, tanh_activation, DELTA_T);

    const auto x_t = torch::randn({batch_size, neuron_number}, torch::requires_grad());
    const auto input_t = torch::randn({batch_size, input_size}, torch::requires_grad());

    liquid_cell.forward(x_t, input_t).sum().backward();

    assert_all_parameters_receive_gradient(liquid_cell);

    ASSERT_TRUE(x_t.grad().defined());
    ASSERT_GT(x_t.grad().abs().sum().item<float>(), 0.f);
    ASSERT_TRUE(input_t.grad().defined());
    ASSERT_GT(input_t.grad().abs().sum().item<float>(), 0.f);
}

INSTANTIATE_TEST_SUITE_P(
    TestLiquidCell, LiquidCellTestParam,
    testing::Combine(
        testing::Values(2, 8, 16), testing::Values(1, 3, 12), testing::Values(1, 2, 6),
        testing::Values(1, 4)));

/*
 * Liquid recurrent
 */

TEST_P(LiquidRecurrentTestParam, OutputShape) {
    const auto [neuron_number, input_size, output_size, unfolding_steps, batch_size, time_steps] =
        GetParam();

    LiquidRecurrent liquid_recurrent(
        neuron_number, input_size, output_size, unfolding_steps, tanh_activation, DELTA_T);

    const auto inputs = torch::randn({batch_size, time_steps, input_size});

    const auto output = liquid_recurrent.forward(inputs);

    ASSERT_EQ(output.ndimension(), 3);
    ASSERT_EQ(output.size(0), batch_size);
    ASSERT_EQ(output.size(1), time_steps);
    ASSERT_EQ(output.size(2), output_size);
}

TEST_P(LiquidRecurrentTestParam, GradientFlows) {
    const auto [neuron_number, input_size, output_size, unfolding_steps, batch_size, time_steps] =
        GetParam();

    // LayerNorm over a single feature outputs exactly zero (x - mean(x) == 0),
    // which blocks any gradient from reaching the layers before it
    if (output_size == 1) GTEST_SKIP() << "LayerNorm({1}) zeroes the signal, no gradient flows";

    LiquidRecurrent liquid_recurrent(
        neuron_number, input_size, output_size, unfolding_steps, tanh_activation, DELTA_T);

    const auto inputs = torch::randn({batch_size, time_steps, input_size}, torch::requires_grad());

    liquid_recurrent.forward(inputs).sum().backward();

    assert_all_parameters_receive_gradient(liquid_recurrent);

    ASSERT_TRUE(inputs.grad().defined());
    ASSERT_TRUE(torch::all(torch::isfinite(inputs.grad())).item<bool>());
    // every time step must contribute to the loss
    const auto per_step_grad = inputs.grad().abs().sum(std::vector<int64_t>{0, 2});
    ASSERT_TRUE(torch::all(torch::gt(per_step_grad, 0.f)).item<bool>())
        << "Some time steps received no gradient: " << per_step_grad;
}

TEST(LiquidRecurrentEdge, RejectsNon3DInput) {
    LiquidRecurrent liquid_recurrent(4, 3, 2, 2, tanh_activation, DELTA_T);

    EXPECT_THROW(liquid_recurrent.forward(torch::randn({2, 3})), c10::Error);
    EXPECT_THROW(liquid_recurrent.forward(torch::randn({2, 5, 3, 1})), c10::Error);
}

INSTANTIATE_TEST_SUITE_P(
    TestLiquidRecurrent, LiquidRecurrentTestParam,
    testing::Combine(
        testing::Values(2, 8, 16), testing::Values(1, 3, 12), testing::Values(1, 2, 6),
        testing::Values(1, 3), testing::Values(1, 4), testing::Values(1, 2, 5)));
