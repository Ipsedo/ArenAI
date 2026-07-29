//
// Created by samuel on 30/06/2026.
//

#include <networks/vision.h>

#include <arenai_agent_tests/tests_networks/tests_vision.h>

using namespace arenai;
using namespace arenai::agent;

TEST_P(VisionTestParam, TestVisionForward) {
    const auto [width, height, channels, output_conv_channels, group_norm_nums, batch_size] =
        GetParam();

    std::vector<std::tuple<int, int>> conv_layers;

    int curr_channels = channels;
    for (const auto &c_o: output_conv_channels) {
        conv_layers.emplace_back(curr_channels, c_o);
        curr_channels = c_o;
    }

    ConvolutionNetwork conv(height, width, conv_layers, group_norm_nums);

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
    TestVision, VisionTestParam,
    testing::Combine(
        testing::Values(16, 32), testing::Values(16, 32), testing::Values(1, 2, 3),
        testing::Values(
            OutputConvChannels{4}, OutputConvChannels{4, 8}, OutputConvChannels{16, 32, 48}),
        testing::Values(GroupNormNums{2, 4, 8}), testing::Values(1, 2, 3)));
