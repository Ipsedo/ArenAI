#ifndef ARENAI_PANORAMA_MASK_NETWORK_H
#define ARENAI_PANORAMA_MASK_NETWORK_H

#include <tuple>
#include <vector>

#include <torch/torch.h>

namespace arenai::agent {

    // predicts, per bearing column of the full-turn panorama, whether an enemy is visible there
    class PanoramaMaskNetwork final : public torch::nn::Module {
    public:
        PanoramaMaskNetwork(
            const std::vector<std::tuple<int, int>> &channels,
            const std::vector<int> &group_norm_nums, int output_channels);

        // [B, 3, PANORAMA_HEIGHT, PANORAMA_WIDTH] uint8 -> [B, columns] logits
        torch::Tensor logits(const torch::Tensor &panorama);

        // -> [B, columns] detached probabilities
        torch::Tensor forward(const torch::Tensor &panorama);

        int get_output_size() const;

    private:
        torch::nn::ModuleList convolutions;
        torch::nn::ModuleList norms;
        torch::nn::Conv2d to_columns{nullptr};
        torch::nn::Conv1d to_logits{nullptr};
        int output_size;
    };

}// namespace arenai::agent

#endif//ARENAI_PANORAMA_MASK_NETWORK_H
