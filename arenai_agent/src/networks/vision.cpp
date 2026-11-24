#include "./vision.h"

using namespace arenai;
using namespace arenai::agent;

namespace {
    constexpr int PADDING = 1, STRIDE = 2, KERNEL = 3;

    int convolved(const int size) { return (size - KERNEL + 2 * PADDING) / STRIDE + 1; }
}// namespace

namespace arenai::agent {
    std::tuple<int, int>
    presence_grid_size(const int vision_height, const int vision_width, const int nb_blocks) {
        int rows = vision_height, cols = vision_width;
        for (int i = 0; i < nb_blocks; i++) {
            rows = convolved(rows);
            cols = convolved(cols);
        }
        return {rows, cols};
    }

    ConvolutionNetwork::ConvolutionNetwork(
        const int vision_height, const int vision_width,
        const std::vector<std::tuple<int, int>> &channels, const std::vector<int> &group_norm_nums)
        : cnn(register_module("cnn", torch::nn::Sequential())) {

        const auto [h, w] =
            presence_grid_size(vision_height, vision_width, static_cast<int>(channels.size()));

        for (int i = 0; i < channels.size(); i++) {
            const auto &[c_i, c_o] = channels[i];
            const auto groups = group_norm_nums[i];

            cnn->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c_i, c_o, KERNEL)
                                                 .stride(STRIDE)
                                                 .padding(PADDING)
                                                 .bias(false)));
            cnn->push_back(torch::nn::GroupNorm(torch::nn::GroupNormOptions(groups, c_o)));
            cnn->push_back(torch::nn::SiLU());
        }

        const int out_channels = std::get<1>(channels.back());

        output_size = 3 * out_channels;

        presence = register_module(
            "presence", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, 1, 1)));
        log_temperature = register_parameter("log_temperature", torch::zeros({out_channels}));

        const auto ys = torch::linspace(-1.f, 1.f, h).unsqueeze(1).expand({h, w});
        const auto xs = torch::linspace(-1.f, 1.f, w).unsqueeze(0).expand({h, w});
        // (2, H * W)
        coordinates = register_buffer("coordinates", torch::stack({xs, ys}).flatten(1));
    }

    torch::Tensor ConvolutionNetwork::normalized(const torch::Tensor &input) {
        TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be UInt8");

        return input.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f);
    }

    VisionEncoding ConvolutionNetwork::encode(const torch::Tensor &input) {
        const auto map = cnn->forward(normalized(input));
        // (B, C, H * W)
        const auto features = map.flatten(2);
        const auto attention =
            torch::softmax(features * torch::exp(-log_temperature).unsqueeze(1), -1);

        // (B, C, 2)
        const auto keypoints = torch::matmul(attention, coordinates.t());
        // (B, C, 1)
        const auto amplitude = (attention * features).sum(-1, true);

        return {.map = map, .keypoints = torch::cat({keypoints, amplitude}, -1).flatten(1)};
    }

    torch::Tensor ConvolutionNetwork::presence_logits(const torch::Tensor &input) {
        return presence->forward(cnn->forward(normalized(input))).squeeze(1);
    }

    std::vector<torch::Tensor> ConvolutionNetwork::presence_parameters() const {
        auto parameters = cnn->parameters();
        for (const auto &parameter: presence->parameters()) parameters.push_back(parameter);
        return parameters;
    }

    int ConvolutionNetwork::get_output_size() const { return output_size; }
}// namespace arenai::agent
