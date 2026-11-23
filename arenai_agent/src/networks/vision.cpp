#include "./vision.h"

#include <algorithm>

#include "./constants.h"

using namespace arenai;
using namespace arenai::agent;

namespace {
    constexpr int PADDING = 1, STRIDE = 2, KERNEL = 3;
    constexpr int MODULES_PER_BLOCK = 3;

    int convolved(const int size) { return (size - KERNEL + 2 * PADDING) / STRIDE + 1; }
}// namespace

namespace arenai::agent {
    std::tuple<int, int>
    presence_grid_size(const int vision_height, const int vision_width, const int nb_blocks) {
        int rows = vision_height, cols = vision_width;
        for (int i = 0; i < std::min(PRESENCE_BLOCKS, nb_blocks); i++) {
            rows = convolved(rows);
            cols = convolved(cols);
        }
        return {rows, cols};
    }

    ConvolutionNetwork::ConvolutionNetwork(
        const int vision_height, const int vision_width,
        const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums)
        : cnn(register_module("cnn", torch::nn::Sequential())),
          nb_presence_modules(
              MODULES_PER_BLOCK * std::min(PRESENCE_BLOCKS, static_cast<int>(channels.size()))) {

        int w = vision_width, h = vision_height;

        for (int i = 0; i < channels.size(); i++) {
            const auto &[c_i, c_o] = channels[i];
            const auto groups = group_norm_nums[i];

            w = convolved(w);
            h = convolved(h);

            cnn->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c_i, c_o, KERNEL)
                                                 .stride(STRIDE)
                                                 .padding(PADDING)
                                                 .bias(false)));
            cnn->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, c_o)));
            cnn->push_back(torch::nn::SiLU());
        }

        output_size = w * h * std::get<1>(channels.back());

        cnn->push_back(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1).end_dim(-1)));

        const int presence_channels =
            std::get<1>(channels[nb_presence_modules / MODULES_PER_BLOCK - 1]);
        presence = register_module(
            "presence", torch::nn::Conv2d(torch::nn::Conv2dOptions(presence_channels, 1, 1)));
    }

    torch::Tensor ConvolutionNetwork::normalized(const torch::Tensor &input) {
        TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be UInt8");

        return input.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f);
    }

    torch::Tensor ConvolutionNetwork::forward(const torch::Tensor &input) {
        return cnn->forward(normalized(input));
    }

    torch::Tensor ConvolutionNetwork::presence_logits(const torch::Tensor &input) {
        auto features = normalized(input);

        auto module = cnn->begin();
        for (int i = 0; i < nb_presence_modules; i++, ++module)
            features = module->forward(features);

        return presence->forward(features).squeeze(1);
    }

    std::vector<torch::Tensor> ConvolutionNetwork::presence_parameters() const {
        auto parameters = presence->parameters();

        auto module = cnn->begin();
        for (int i = 0; i < nb_presence_modules; i++, ++module)
            for (const auto &parameter: module->ptr()->parameters())
                parameters.push_back(parameter);

        return parameters;
    }

    int ConvolutionNetwork::get_output_size() const { return output_size; }
}// namespace arenai::agent
