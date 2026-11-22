#include <cmath>

#include <agents/ppo_objective.h>
#include <gtest/gtest.h>

using namespace arenai;
using namespace arenai::agent;

TEST(PpoObjective, GeneralizedAdvantagesStopAtACut) {
    // [T = 3, nb_tanks = 1, 1], the episode cut after the second step
    const auto deltas = torch::tensor({1.f, 2.f, 4.f}).reshape({3, 1, 1});
    const auto cuts = torch::tensor({0.f, 1.f, 0.f}).reshape({3, 1, 1});

    const auto advantages = generalized_advantages(deltas, cuts, 0.5f);

    const auto expected = torch::tensor({1.f + 0.5f * 2.f, 2.f, 4.f}).reshape({3, 1, 1});
    ASSERT_TRUE(torch::allclose(advantages, expected));
}

TEST(PpoObjective, NormalizedIgnoresDeadTransitions) {
    const auto advantages = torch::tensor({1.f, 3.f, 100.f}).reshape({3, 1, 1});
    const auto valids = torch::tensor({1.f, 1.f, 0.f}).reshape({3, 1, 1});

    const auto result = normalized(advantages, valids);

    ASSERT_NEAR(result[0].item<float>(), -1.f, 1e-5f);
    ASSERT_NEAR(result[1].item<float>(), 1.f, 1e-5f);
}

TEST(PpoObjective, SurrogateLearnsFromTheAdvantage) {
    const auto log_probs = torch::zeros({3, 1}, torch::requires_grad());
    const auto advantages = torch::tensor({-4.f, 5.f, 0.f}).reshape({3, 1});

    const auto [surrogate, log_ratio, ratio, kl] =
        clipped_surrogate(log_probs, torch::zeros({3, 1}), advantages, 0.2f);

    ASSERT_TRUE(torch::allclose(surrogate, advantages));
    ASSERT_TRUE(torch::allclose(ratio, torch::ones({3, 1})));

    surrogate.sum().backward();
    // at ratio 1, d(surrogate)/d(log-prob) is the advantage
    ASSERT_TRUE(torch::allclose(log_probs.grad(), advantages));
}

TEST(PpoObjective, SurrogateIsClipped) {
    const auto [surrogate, log_ratio, ratio, kl] = clipped_surrogate(
        torch::full({1, 1}, std::log(2.f)), torch::zeros({1, 1}), torch::ones({1, 1}), 0.2f);

    ASSERT_NEAR(surrogate.item<float>(), 1.2f, 1e-5f);
    ASSERT_NEAR(ratio.item<float>(), 2.f, 1e-5f);
    // r - 1 - log r
    ASSERT_NEAR(kl.item<float>(), 1.f - std::log(2.f), 1e-5f);
}

TEST(PpoObjective, KlStopStopsPastTheThreshold) {
    KlStop kl_stop(0.05f);
    kl_stop.on_planned_steps(4);

    ASSERT_FALSE(kl_stop.stop_exceeded(0.07f));
    kl_stop.on_step();

    // past 1.5 * 0.05
    ASSERT_TRUE(kl_stop.stop_exceeded(0.08f));
    ASSERT_TRUE(kl_stop.stopped());

    ASSERT_FLOAT_EQ(kl_stop.skip_fraction(), 1.f - 1.f / 4.f);
}

TEST(PpoObjective, KlStopNeverStopsWithoutTarget) {
    KlStop kl_stop(0.f);

    ASSERT_FALSE(kl_stop.stop_exceeded(100.f));
    ASSERT_FALSE(kl_stop.stopped());
    ASSERT_FLOAT_EQ(kl_stop.skip_fraction(), 0.f);
}

TEST(PpoObjective, EpisodeEndTakesItsClosingValueAndStopsTheRecursion) {
    // [T = 3, nb_tanks = 2, 1], the episode ends after the second step
    const auto next_values = torch::zeros({3, 2, 1});
    const auto episode_ends = torch::tensor({false, true, false});
    const auto closing_value = torch::tensor({5.f, 7.f}).reshape({2, 1});

    const auto closed = closed_next_values(next_values, episode_ends, {closing_value});

    ASSERT_TRUE(torch::allclose(closed[1], closing_value));
    ASSERT_FLOAT_EQ(closed[0].abs().sum().item<float>(), 0.f);
    ASSERT_FLOAT_EQ(closed[2].abs().sum().item<float>(), 0.f);

    const auto cuts = torch::zeros({3, 2, 1});
    const auto segment_ends = with_episode_ends(cuts, episode_ends);
    const auto deltas = torch::ones({3, 2, 1});

    const auto advantages = generalized_advantages(deltas, segment_ends, 0.5f);

    ASSERT_FLOAT_EQ(advantages[1][0].item<float>(), 1.f);
    ASSERT_FLOAT_EQ(advantages[0][0].item<float>(), 1.5f);
}
