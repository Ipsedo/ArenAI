//
// Created by samuel on 06/09/2026.
//

#include "./liquid_recurrent.h"

#include "../../networks_utils/init.h"

using namespace arenai;
using namespace arenai::agent;

/*
 * Cell model
 */

CellModel::CellModel(
    const int neuron_number, const int input_size,
    const std::function<torch::Tensor(const torch::Tensor &)> &activation_function)
    : weights(register_module(
        "weights",
        torch::nn::Linear(torch::nn::LinearOptions(input_size, neuron_number).bias(false)))),
      recurrent_weights(register_module(
          "recurrent_weights",
          torch::nn::Linear(torch::nn::LinearOptions(neuron_number, neuron_number).bias(false)))),
      biases(register_parameter("biases", torch::zeros({1, neuron_number}))),
      activation_function(activation_function) {

    weights->apply(init_liquid_weights);
    recurrent_weights->apply(init_liquid_weights);
}

torch::Tensor CellModel::forward(const torch::Tensor &x_t, const torch::Tensor &input_t) {
    return activation_function(recurrent_weights(x_t) + weights(input_t) + biases);
}

/*
 * Liquid cell
 */
LiquidCell::LiquidCell(
    const int neuron_number, const int input_size, const int unfolding_steps,
    const std::function<torch::Tensor(const torch::Tensor &)> &activation_function,
    const float delta_t)
    : a(register_parameter("a", torch::ones({1, neuron_number}))),
      raw_tau(register_parameter("raw_tau", torch::zeros({1, neuron_number}))),
      f(register_module(
          "f", std::make_shared<CellModel>(neuron_number, input_size, activation_function))),
      unfolding_steps(unfolding_steps), delta_t(delta_t) {}

torch::Tensor LiquidCell::forward(const torch::Tensor &x_t, const torch::Tensor &input_t) {
    auto x_t_next = x_t;
    const auto curr_delta_t = delta_t / static_cast<float>(unfolding_steps);

    for (int i = 0; i < unfolding_steps; ++i) {
        const auto f_output = f->forward(x_t_next, input_t);
        x_t_next = (x_t_next + curr_delta_t * f_output * a)
                   / (1.0 + curr_delta_t * (1.0 / tau() + f_output));
    }

    return x_t_next;
}

torch::Tensor LiquidCell::tau() const { return torch::nn::functional::softplus(raw_tau) + 1e-3; }

/*
 * Liquid recurrent
 */

LiquidRecurrent::LiquidRecurrent(
    int neuron_number, int input_size, int output_size, int unfolding_steps,
    const std::function<torch::Tensor(const torch::Tensor &)> &activation_function, float delta_t)
    : cell(register_module(
        "cell", std::make_shared<LiquidCell>(
                    neuron_number, input_size, unfolding_steps, activation_function, delta_t))),
      neuron_number(neuron_number),
      to_output(register_module(
          "to_output",
          torch::nn::Sequential(
              torch::nn::Linear(torch::nn::LinearOptions(neuron_number, output_size).bias(false)),
              torch::nn::LayerNorm(
                  torch::nn::LayerNormOptions({output_size}).elementwise_affine(true)),
              torch::nn::SiLU()))) {}

torch::Tensor LiquidRecurrent::forward(const torch::Tensor &inputs) {
    TORCH_CHECK(
        inputs.sizes().size() == 3,
        "Processed input needs to have 3 dimensions (Batch, Time, Features)");

    const auto batch_size = inputs.size(0);
    const auto nb_steps = inputs.size(1);

    auto x_t = get_first_x(static_cast<int>(batch_size));

    std::vector<torch::Tensor> results;
    results.reserve(nb_steps);

    for (auto t = 0; t < nb_steps; t++) {
        auto [output, x_t_next] =
            forward_step(x_t, inputs.index({at::indexing::Slice(), t, at::indexing::Slice()}));

        x_t = x_t_next;
        results.push_back(output);
    }

    // (batch, time, output_features)
    return torch::stack(results, 1);
}

std::tuple<torch::Tensor, torch::Tensor>
LiquidRecurrent::forward_step(const torch::Tensor &x_t, const torch::Tensor &input_t) {
    const auto x_t_next = cell->forward(x_t, input_t);
    return {to_output->forward(x_t_next), x_t_next};
}

torch::Tensor LiquidRecurrent::get_first_x(int batch_size) {
    return 1e-1 * torch::randn({batch_size, neuron_number}, parameters().back().device());
}
