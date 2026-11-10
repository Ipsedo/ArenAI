#include <core/potential_shaping.h>
#include <gtest/gtest.h>

using namespace arenai;
using namespace arenai::agent;

namespace {
    constexpr float GAMMA = 0.9f;
    constexpr float SCALE = 0.1f;
}// namespace

TEST(PotentialShaping, PaysTheDiscountedDifference) {
    PotentialShaping shaping(GAMMA, SCALE, 2);
    shaping.reset({0.2f, 0.5f});

    const auto rewards = shaping.step({0.6f, 0.5f}, {false, false});

    ASSERT_NEAR(rewards[0], SCALE * (GAMMA * 0.6f - 0.2f), 1e-6f);
    ASSERT_NEAR(rewards[1], SCALE * (GAMMA * 0.5f - 0.5f), 1e-6f);
}

TEST(PotentialShaping, TerminalStateHasZeroPotential) {
    PotentialShaping shaping(GAMMA, SCALE, 1);
    shaping.reset({0.8f});

    const auto rewards = shaping.step({0.9f}, {true});
    ASSERT_NEAR(rewards[0], -SCALE * 0.8f, 1e-6f);

    const auto after = shaping.step({0.9f}, {true});
    ASSERT_FLOAT_EQ(after[0], 0.f);
}

TEST(PotentialShaping, SumOverAnEpisodeTelescopesToMinusTheInitialPotential) {
    PotentialShaping shaping(GAMMA, SCALE, 1);
    shaping.reset({0.3f});

    const std::vector potentials{0.1f, 0.7f, 0.4f, 0.9f};
    float discounted_sum = 0.f, discount = 1.f;
    for (size_t t = 0; t < potentials.size(); t++) {
        const bool last = t + 1 == potentials.size();
        discounted_sum += discount * shaping.step({potentials[t]}, {last})[0];
        discount *= GAMMA;
    }

    ASSERT_NEAR(discounted_sum, -SCALE * 0.3f, 1e-6f);
}
