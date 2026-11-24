#ifndef ARENAI_AGENT_HOST_VISION_H
#define ARENAI_AGENT_HOST_VISION_H

#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "./presence_model.h"

namespace arenai::agent {

    class ConvolutionNetwork final : public torch::nn::Module, public PresenceModel {
    public:
        explicit ConvolutionNetwork(
            int vision_height, int vision_width, const std::vector<std::tuple<int, int>> &channels,
            const std::vector<int> &group_norm_nums);

        torch::Tensor forward(const torch::Tensor &input);

        torch::Tensor presence_logits(const torch::Tensor &input) override;
        // the convolution blocks and the head
        std::vector<torch::Tensor> presence_parameters() const override;

        int get_output_size() const;

    private:
        torch::nn::Sequential cnn{nullptr};
        torch::nn::Conv2d presence{nullptr};
        torch::Tensor log_temperature;
        torch::Tensor coordinates;
        int output_size;

        static torch::Tensor normalized(const torch::Tensor &input);
    };

    // (rows, cols) of the presence head of a network with nb_blocks convolution blocks
    std::tuple<int, int> presence_grid_size(int vision_height, int vision_width, int nb_blocks);

}// namespace arenai::agent

#endif// ARENAI_AGENT_HOST_VISION_H
