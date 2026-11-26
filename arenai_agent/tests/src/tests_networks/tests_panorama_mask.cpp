#include <gtest/gtest.h>
#include <networks/panorama_mask.h>

#include <arenai_core/types.h>

using namespace arenai;
using namespace arenai::agent;

namespace {
    PanoramaMaskNetwork make_network() {
        return PanoramaMaskNetwork({{3, 8}, {8, 16}, {16, 16}}, {2, 4, 4}, 4);
    }

    torch::Tensor random_panorama(const int batch_size) {
        return torch::randint(
            255, {batch_size, 3, core::PANORAMA_HEIGHT, core::PANORAMA_WIDTH}, torch::kUInt8);
    }
}// namespace

TEST(PanoramaMaskNetworkTest, OutputIsOneProbabilityPerColumn) {
    auto network = make_network();

    const auto output = network.forward(random_panorama(3));

    // three stride-2 blocks: one column every 8 pixels
    ASSERT_EQ(network.get_output_size(), core::PANORAMA_WIDTH / 8);
    ASSERT_EQ(output.sizes().vec(), (std::vector<int64_t>{3, network.get_output_size()}));
    ASSERT_TRUE(torch::all(output.ge(0.f).logical_and(output.le(1.f))).item<bool>());
}

TEST(PanoramaMaskNetworkTest, ProbabilitiesAreTheSigmoidOfTheLogits) {
    auto network = make_network();
    const auto panorama = random_panorama(2);

    ASSERT_TRUE(torch::allclose(
        network.forward(panorama), torch::sigmoid(network.logits(panorama)), 1e-5, 1e-6));
}

TEST(PanoramaMaskNetworkTest, ProbabilitiesCarryNoGradient) {
    auto network = make_network();
    const auto panorama = random_panorama(2);

    ASSERT_FALSE(network.forward(panorama).requires_grad());
    ASSERT_TRUE(network.logits(panorama).requires_grad());
}

TEST(PanoramaMaskNetworkTest, RejectsNonUint8Input) {
    auto network = make_network();

    ASSERT_THROW(
        network.forward(torch::randn({1, 3, core::PANORAMA_HEIGHT, core::PANORAMA_WIDTH})),
        c10::Error);
}

TEST(PanoramaMaskNetworkTest, TurningThePanoramaTurnsTheColumns) {
    auto network = make_network();
    network.eval();

    const auto panorama = random_panorama(2);

    torch::NoGradGuard no_grad;
    const auto output = network.logits(panorama);
    // 24 pixels = 3 output columns: the wrap-around border must behave like any other column
    const auto turned = network.logits(panorama.roll(24, 3));

    ASSERT_TRUE(torch::allclose(turned, output.roll(3, 1), 1e-4, 1e-5));
}
