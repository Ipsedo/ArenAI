//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_LIQUID_RECURRENT_H
#define ARENAI_LIQUID_RECURRENT_H

#include <torch/torch.h>

namespace arenai::agent {
    class CellModel : public torch::nn::Module {
    public:
        CellModel(
            int neuron_number, int input_size,
            const std::function<torch::Tensor(const torch::Tensor &)> &activation_function);

        torch::Tensor forward(const torch::Tensor &x_t, const torch::Tensor &input_t);

    private:
        torch::nn::Linear weights;
        torch::nn::Linear recurrent_weights;
        torch::Tensor biases;

        std::function<torch::Tensor(const torch::Tensor &)> activation_function;
    };

    class LiquidCell : public torch::nn::Module {
    public:
        LiquidCell(
            int neuron_number, int input_size, int unfolding_steps,
            const std::function<torch::Tensor(const torch::Tensor &)> &activation_function,
            float delta_t);

        torch::Tensor forward(const torch::Tensor &x_t, const torch::Tensor &input_t);

    private:
        torch::Tensor a;
        torch::Tensor raw_tau;

        std::shared_ptr<CellModel> f;

        int unfolding_steps;
        float delta_t;

        torch::Tensor tau() const;
    };

    class LiquidRecurrent : public torch::nn::Module {
    public:
        LiquidRecurrent(
            int neuron_number, int input_size, int output_size, int unfolding_steps,
            const std::function<torch::Tensor(const torch::Tensor &)> &activation_function,
            float delta_t);

        torch::Tensor forward(const torch::Tensor &inputs);

        // one recurrent step, the state is handled by the caller: (output, x_t_next)
        std::tuple<torch::Tensor, torch::Tensor>
        forward_step(const torch::Tensor &x_t, const torch::Tensor &input_t);

        torch::Tensor get_first_x(int batch_size);

    private:
        std::shared_ptr<LiquidCell> cell;
        int neuron_number;

        torch::nn::Sequential to_output;
    };
}// namespace arenai::agent

#endif//ARENAI_LIQUID_RECURRENT_H
