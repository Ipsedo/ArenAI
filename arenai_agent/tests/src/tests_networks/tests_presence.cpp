#include <agents/presence_learner.h>
#include <gtest/gtest.h>
#include <metrics/mean_metric.h>
#include <networks/constants.h>
#include <networks/vision.h>

using namespace arenai;
using namespace arenai::agent;

TEST(PresenceTest, GridSizeFollowsTheConvolutionBlocks) {
    const auto [rows, cols] = presence_grid_size(128, 256, 7);
    ASSERT_EQ(rows, 128 >> PRESENCE_BLOCKS);
    ASSERT_EQ(cols, 256 >> PRESENCE_BLOCKS);

    const auto [shallow_rows, shallow_cols] = presence_grid_size(8, 16, 1);
    ASSERT_EQ(shallow_rows, 4);
    ASSERT_EQ(shallow_cols, 8);
}

TEST(PresenceTest, LogitsHaveTheGridShape) {
    const std::vector<std::tuple<int, int>> channels{{3, 4}, {4, 4}, {4, 8}, {8, 8}, {8, 8}};
    const auto conv =
        std::make_shared<ConvolutionNetwork>(32, 64, channels, std::vector{2, 2, 2, 2, 2});

    const auto [rows, cols] = presence_grid_size(32, 64, static_cast<int>(channels.size()));
    const auto logits = conv->presence_logits(torch::zeros({3, 3, 32, 64}, torch::kUInt8));

    ASSERT_EQ(logits.sizes(), (std::vector<int64_t>{3, rows, cols}));
}

TEST(PresenceTest, FitLowersTheLossThroughTheFirstBlocksOnly) {
    torch::manual_seed(0);

    const std::vector<std::tuple<int, int>> channels{{3, 4}, {4, 4}, {4, 8}, {8, 8}, {8, 8}};
    const auto conv =
        std::make_shared<ConvolutionNetwork>(32, 64, channels, std::vector{2, 2, 2, 2, 2});
    const PresenceLearner learner(conv, 1e-2f, std::make_shared<MeanMetric>("pres", 8));

    constexpr int nb_rows = 16;
    const auto [rows, cols] = presence_grid_size(32, 64, static_cast<int>(channels.size()));
    const auto vision = torch::randint(0, 255, {nb_rows, 3, 32, 64}, torch::kUInt8);
    const auto presence = torch::randint(0, 2, {nb_rows, rows, cols}).to(torch::kFloat);
    const auto idx = torch::arange(nb_rows);

    const auto loss = [&] {
        torch::NoGradGuard no_grad;
        return torch::binary_cross_entropy_with_logits(conv->presence_logits(vision), presence)
            .item<float>();
    };

    const auto named = conv->named_parameters();
    const auto last_block = named["cnn.12.weight"].detach().clone();
    const auto first_block = named["cnn.0.weight"].detach().clone();

    const float before = loss();
    for (int i = 0; i < 50; i++) learner.fit(vision, presence, idx, 8);

    ASSERT_LT(loss(), before);
    ASSERT_FALSE(torch::equal(named["cnn.0.weight"], first_block));
    ASSERT_TRUE(torch::equal(named["cnn.12.weight"], last_block));
}
