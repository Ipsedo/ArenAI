//
// Created by claude on 20/09/2026.
//

#include <networks/vision.h>

#include <arenai_agent_tests/tests_networks/tests_vision_impala.h>

using namespace arenai;
using namespace arenai::agent;

TEST_P(ImpalaVisionTestParam, TestImpalaVisionForward) {
    const auto [width, height, channels, output_conv_channels, batch_size] = GetParam();

    std::vector<std::tuple<int, int>> conv_layers;

    int curr_channels = channels;
    for (const auto &c_o: output_conv_channels) {
        conv_layers.emplace_back(curr_channels, c_o);
        curr_channels = c_o;
    }

    ImpalaConvolutionNetwork conv(height, width, conv_layers);

    const auto images = torch::randint(
        255, {batch_size, channels, height, width}, torch::TensorOptions().dtype(torch::kUInt8));

    const auto encoded_images = conv.forward(images);

    ASSERT_EQ(encoded_images.ndimension(), 2);
    ASSERT_EQ(encoded_images.size(0), batch_size);
    ASSERT_EQ(encoded_images.size(1), conv.get_output_size());
    ASSERT_EQ(encoded_images.size(1) % output_conv_channels.back(), 0);
}

// Create parametrized tests

INSTANTIATE_TEST_SUITE_P(
    TestImpalaVision, ImpalaVisionTestParam,
    testing::Combine(
        testing::Values(16, 32), testing::Values(16, 32), testing::Values(1, 2, 3),
        testing::Values(
            ImpalaOutputConvChannels{4}, ImpalaOutputConvChannels{4, 8},
            ImpalaOutputConvChannels{16, 32, 48}),
        testing::Values(1, 2, 3)));

// Edge cases

TEST_F(ImpalaVisionEdgeTest, RejectsNonUint8Input) {
    ImpalaConvolutionNetwork conv(8, 8, {{3, 4}});

    const auto float_input = torch::randn({1, 3, 8, 8});

    ASSERT_THROW(conv.forward(float_input), c10::Error) << "Should throw when input is not UInt8";
}

TEST_F(ImpalaVisionEdgeTest, NormalizesToExpectedRange) {
    ImpalaConvolutionNetwork conv(8, 8, {{3, 4}});

    const auto zeros = torch::zeros({1, 3, 8, 8}, torch::kUInt8);
    const auto result_zeros = conv.forward(zeros);

    const auto max255 = torch::ones({1, 3, 8, 8}, torch::kUInt8) * 255;
    const auto result_255 = conv.forward(max255);

    ASSERT_TRUE(torch::all(torch::isfinite(result_zeros)).item<bool>());
    ASSERT_TRUE(torch::all(torch::isfinite(result_255)).item<bool>());
}

TEST_F(ImpalaVisionEdgeTest, OutputSizeMatchesGetOutputSize) {
    const std::vector<std::tuple<int, int>> channels = {{3, 8}, {8, 16}};
    constexpr int h = 16, w = 16;
    constexpr int batch_size = 2;

    ImpalaConvolutionNetwork conv(h, w, channels);

    const auto input = torch::randint(255, {batch_size, 3, h, w}, torch::kUInt8);
    const auto output = conv.forward(input);

    ASSERT_EQ(output.size(0), batch_size);
    ASSERT_EQ(output.size(1), conv.get_output_size());
}

TEST_F(ImpalaVisionEdgeTest, ResidualBlockPreservesShape) {
    ResidualBlock block(8);

    const auto input = torch::randn({2, 8, 16, 16});
    const auto output = block.forward(input);

    ASSERT_TRUE(output.sizes() == input.sizes());
    ASSERT_TRUE(torch::all(torch::isfinite(output)).item<bool>());
}
