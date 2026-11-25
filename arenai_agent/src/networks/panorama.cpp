#include "./panorama.h"

#include <arenai_core/types.h>

namespace arenai::agent {

    namespace F = torch::nn::functional;

    PanoramaNetwork::PanoramaNetwork(
        const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums,
        const int output_channels)
        : convolutions(register_module("convolutions", torch::nn::ModuleList())),
          norms(register_module("norms", torch::nn::ModuleList())) {

        int w = core::PANORAMA_WIDTH;

        for (int i = 0; i < static_cast<int>(channels.size()); i++) {
            const auto &[c_i, c_o] = channels[i];

            // the padding is applied by hand in forward: circular in width, zero in height
            convolutions->push_back(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(c_i, c_o, 3).stride(2).bias(false)));
            norms->push_back(
                torch::nn::GroupNorm(torch::nn::GroupNormOptions(group_norm_nums[i], c_o)));

            w = (w - 1) / 2 + 1;
        }

        to_columns = register_module(
            "to_columns", torch::nn::Conv2d(torch::nn::Conv2dOptions(
                              std::get<1>(channels.back()), output_channels, 1)));

        output_size = w * output_channels;
    }

    torch::Tensor PanoramaNetwork::forward(const torch::Tensor &input) {
        TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be UInt8");

        auto out = input.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f);

        for (size_t i = 0; i < convolutions->size(); i++) {
            out = F::pad(out, F::PadFuncOptions({1, 1, 0, 0}).mode(torch::kCircular));
            out = F::pad(out, F::PadFuncOptions({0, 0, 1, 1}));
            out = convolutions[i]->as<torch::nn::Conv2d>()->forward(out);
            out = torch::silu(norms[i]->as<torch::nn::GroupNorm>()->forward(out));
        }

        // [B, output_channels, rows, columns] -> strongest row per column -> [B, output_size]
        return to_columns->forward(out).amax(2).flatten(1);
    }

    int PanoramaNetwork::get_output_size() const { return output_size; }

}// namespace arenai::agent
