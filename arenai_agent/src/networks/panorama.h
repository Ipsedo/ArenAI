#ifndef ARENAI_PANORAMA_NETWORK_H
#define ARENAI_PANORAMA_NETWORK_H

#include <tuple>
#include <vector>

#include <torch/torch.h>

namespace arenai::agent {

    // encodes the full-turn panorama; its columns wrap around, so do the convolutions
    class PanoramaNetwork final : public torch::nn::Module {
    public:
        PanoramaNetwork(
            const std::vector<std::tuple<int, int>> &channels,
            const std::vector<int> &group_norm_nums, int output_channels);

        // [B, 3, PANORAMA_HEIGHT, PANORAMA_WIDTH] uint8 -> [B, output_size]
        torch::Tensor forward(const torch::Tensor &input);

        int get_output_size() const;

    private:
        torch::nn::ModuleList convolutions;
        torch::nn::ModuleList norms;
        torch::nn::Conv2d to_columns{nullptr};
        int output_size;
    };

}// namespace arenai::agent

#endif//ARENAI_PANORAMA_NETWORK_H
