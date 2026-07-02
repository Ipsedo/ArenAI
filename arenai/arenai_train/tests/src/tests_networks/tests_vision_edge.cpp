//
// Created by claude on 01/07/2026.
//

#include <networks/vision.h>

#include <arenai_train_tests/tests_networks/tests_vision_edge.h>

using namespace arenai;
using namespace arenai::train;

TEST_F(VisionEdgeTest, RejectsNonUint8Input) {
    ConvolutionNetwork conv(8, 8, {{3, 4}}, {2});

    const auto float_input = torch::randn({1, 3, 8, 8});

    ASSERT_THROW(conv.forward(float_input), std::runtime_error)
        << "Should throw when input is not UInt8";
}

TEST_F(VisionEdgeTest, NormalizesToExpectedRange) {
    ConvolutionNetwork conv(8, 8, {{3, 4}}, {2});

    const auto zeros = torch::zeros({1, 3, 8, 8}, torch::kUInt8);
    const auto result_zeros = conv.forward(zeros);

    const auto max255 = torch::ones({1, 3, 8, 8}, torch::kUInt8) * 255;
    const auto result_255 = conv.forward(max255);

    ASSERT_TRUE(torch::all(torch::isfinite(result_zeros)).item<bool>());
    ASSERT_TRUE(torch::all(torch::isfinite(result_255)).item<bool>());
}

TEST_F(VisionEdgeTest, OutputSizeMatchesGetOutputSize) {
    const std::vector<std::tuple<int, int>> channels = {{3, 8}, {8, 16}};
    const std::vector<int> gnums = {4, 4};
    constexpr int h = 16, w = 16;

    ConvolutionNetwork conv(h, w, channels, gnums);

    const auto input = torch::randint(255, {2, 3, h, w}, torch::kUInt8);
    const auto output = conv.forward(input);

    ASSERT_EQ(output.size(1), conv.get_output_size());
}
