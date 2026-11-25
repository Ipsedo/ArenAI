#include <gtest/gtest.h>
#include <networks/panorama.h>

#include <arenai_core/types.h>

using namespace arenai;
using namespace arenai::agent;

namespace {
    constexpr int OUTPUT_CHANNELS = 4;

    PanoramaNetwork make_network() {
        return PanoramaNetwork({{3, 8}, {8, 16}, {16, 16}}, {2, 4, 4}, OUTPUT_CHANNELS);
    }

    torch::Tensor random_panorama(const int batch_size) {
        return torch::randint(
            255, {batch_size, 3, core::PANORAMA_HEIGHT, core::PANORAMA_WIDTH}, torch::kUInt8);
    }
}// namespace

TEST(PanoramaNetworkTest, OutputIsOneRowOfColumnsPerChannel) {
    auto network = make_network();

    const auto output = network.forward(random_panorama(3));

    // three stride-2 blocks: one column every 8 pixels
    ASSERT_EQ(network.get_output_size(), OUTPUT_CHANNELS * core::PANORAMA_WIDTH / 8);
    ASSERT_EQ(output.sizes().vec(), (std::vector<int64_t>{3, network.get_output_size()}));
    ASSERT_TRUE(torch::all(torch::isfinite(output)).item<bool>());
}

TEST(PanoramaNetworkTest, RejectsNonUint8Input) {
    auto network = make_network();

    ASSERT_THROW(
        network.forward(torch::randn({1, 3, core::PANORAMA_HEIGHT, core::PANORAMA_WIDTH})),
        c10::Error);
}

TEST(PanoramaNetworkTest, TurningThePanoramaTurnsTheColumns) {
    auto network = make_network();
    network.eval();

    const auto panorama = random_panorama(2);
    const auto columns = core::PANORAMA_WIDTH / 8;

    torch::NoGradGuard no_grad;
    const auto output = network.forward(panorama).reshape({2, OUTPUT_CHANNELS, columns});
    // 24 pixels = 3 output columns: the wrap-around border must behave like any other column
    const auto turned =
        network.forward(panorama.roll(24, 3)).reshape({2, OUTPUT_CHANNELS, columns});

    ASSERT_TRUE(torch::allclose(turned, output.roll(3, 2), 1e-4, 1e-5));
}
