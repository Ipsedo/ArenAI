#include "./panorama_mask.h"

#include <arenai_core/types.h>

#include "../networks_utils/init.h"

namespace arenai::agent {

    namespace F = torch::nn::functional;

    PanoramaMaskNetwork::PanoramaMaskNetwork(
        const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums,
        const int output_channels)
        : convolutions(register_module("convolutions", torch::nn::ModuleList())),
          norms(register_module("norms", torch::nn::ModuleList())),
          output_size(core::PANORAMA_WIDTH) {

        for (int i = 0; i < static_cast<int>(channels.size()); i++) {
            const auto &[c_i, c_o] = channels[i];

            // the padding is applied by hand in logits: circular in width, zero in height
            convolutions->push_back(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(c_i, c_o, 3).stride(2).bias(false)));
            norms->push_back(
                torch::nn::GroupNorm(torch::nn::GroupNormOptions(group_norm_nums[i], c_o)));

            output_size = (output_size - 1) / 2 + 1;
        }

        to_columns = register_module(
            "to_columns", torch::nn::Conv2d(torch::nn::Conv2dOptions(
                              std::get<1>(channels.back()), output_channels, 1)));
        to_logits = register_module(
            "to_logits", torch::nn::Conv1d(torch::nn::Conv1dOptions(output_channels, 1, 1)));

        apply(init_hidden_weights);
    }

    torch::Tensor PanoramaMaskNetwork::logits(const torch::Tensor &panorama) {
        TORCH_CHECK(panorama.dtype() == torch::kUInt8, "Input must be UInt8");

        auto out = panorama.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f);

        for (size_t i = 0; i < convolutions->size(); i++) {
            out = F::pad(out, F::PadFuncOptions({1, 1, 0, 0}).mode(torch::kCircular));
            out = F::pad(out, F::PadFuncOptions({0, 0, 1, 1}));
            out = convolutions[i]->as<torch::nn::Conv2d>()->forward(out);
            out = torch::silu(norms[i]->as<torch::nn::GroupNorm>()->forward(out));
        }

        // [B, output_channels, rows, columns] -> mean over rows -> [B, columns]
        return to_logits->forward(to_columns->forward(out).mean(2)).squeeze(1);
    }

    torch::Tensor PanoramaMaskNetwork::forward(const torch::Tensor &panorama) {
        return torch::sigmoid(logits(panorama)).detach();
    }

    int PanoramaMaskNetwork::get_output_size() const { return output_size; }

}// namespace arenai::agent
