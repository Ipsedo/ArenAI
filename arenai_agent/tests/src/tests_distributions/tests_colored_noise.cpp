#include <gtest/gtest.h>

#include "../../../src/distributions/colored_noise.h"

using namespace arenai::agent;

namespace {
    // stacked draws [T, N]
    torch::Tensor draws(ColoredNoise &noise, const int nb_steps, const int nb_series) {
        std::vector<torch::Tensor> steps;
        for (int t = 0; t < nb_steps; t++) steps.push_back(noise.next(torch::zeros({nb_series})));
        return torch::stack(steps);
    }

    float lag_one_correlation(const torch::Tensor &series) {
        const auto head = series.slice(0, 0, series.size(0) - 1);
        const auto tail = series.slice(0, 1);
        return ((head * tail).mean() / series.pow(2).mean()).item<float>();
    }
}// namespace

TEST(ColoredNoiseTest, MarginalIsStandardNormal) {
    torch::manual_seed(0);
    ColoredNoise noise(0.9f);
    const auto series = draws(noise, 400, 2000);

    EXPECT_NEAR(series.mean().item<float>(), 0.f, 0.02f);
    EXPECT_NEAR(series.std().item<float>(), 1.f, 0.02f);
    EXPECT_NEAR(series[399].std().item<float>(), 1.f, 0.06f);
}

TEST(ColoredNoiseTest, LagOneCorrelationIsTheCoefficient) {
    torch::manual_seed(0);
    ColoredNoise noise(0.9f);

    EXPECT_NEAR(lag_one_correlation(draws(noise, 400, 2000)), 0.9f, 0.01f);
}

TEST(ColoredNoiseTest, ZeroCorrelationIsWhite) {
    torch::manual_seed(0);
    ColoredNoise noise(0.f);

    EXPECT_NEAR(lag_one_correlation(draws(noise, 400, 2000)), 0.f, 0.01f);
}

TEST(ColoredNoiseTest, NewShapeRestartsTheSeries) {
    ColoredNoise noise(0.9f);
    noise.next(torch::zeros({4, 2}));

    EXPECT_EQ(noise.next(torch::zeros({7, 2})).size(0), 7);
}

TEST(ColoredNoiseTest, RestartRedrawsOnlyEndedSeries) {
    torch::manual_seed(0);
    ColoredNoise noise(1.f);
    const auto before = noise.next(torch::zeros({3, 2})).clone();

    noise.restart(torch::tensor({false, true, false}).unsqueeze(1));
    const auto after = noise.next(torch::zeros({3, 2}));

    EXPECT_TRUE(torch::equal(after[0], before[0]));
    EXPECT_FALSE(torch::equal(after[1], before[1]));
    EXPECT_TRUE(torch::equal(after[2], before[2]));
}

TEST(ColoredNoiseTest, ResetForgetsTheSeries) {
    torch::manual_seed(0);
    ColoredNoise noise(1.f);
    const auto before = noise.next(torch::zeros({3, 2})).clone();

    noise.reset();

    EXPECT_FALSE(torch::equal(noise.next(torch::zeros({3, 2})), before));
}
