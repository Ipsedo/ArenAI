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

TEST(PpoObjective, ImmediateAimAdvantageIsTheNormalizedAimReward) {
    const auto aim_rewards = torch::randn({5, 4, 1});
    const auto cuts = torch::zeros({5, 4, 1});
    const auto valids = torch::ones({5, 4, 1});

    ASSERT_TRUE(torch::allclose(
        aim_advantages(aim_rewards, cuts, valids, 0.99f, 0.f), normalized(aim_rewards, valids),
        1e-5f, 1e-6f));
}

TEST(PpoObjective, SplitPutsTheCanonActionsInTheAimGroup) {
    // dir_x, dir_y, canon_x, canon_y
    const auto continuous = torch::tensor({{1.f, 2.f, 4.f, 8.f}});
    const auto discrete = torch::tensor({{16.f}});

    const auto [aim, rest] = PolicyLogProbs::split(continuous, discrete);

    ASSERT_FLOAT_EQ(aim.item<float>(), 4.f + 8.f);
    ASSERT_FLOAT_EQ(rest.item<float>(), 1.f + 2.f + 16.f);
}

TEST(PpoObjective, EachGroupLearnsFromItsOwnAdvantage) {
    const auto aim = torch::zeros({3, 1}, torch::requires_grad());
    const auto rest = torch::zeros({3, 1}, torch::requires_grad());
    const PolicyLogProbs old{.aim = torch::zeros({3, 1}), .rest = torch::zeros({3, 1})};

    const auto aim_advantages = torch::tensor({1.f, -2.f, 3.f}).reshape({3, 1});
    const auto advantages = torch::tensor({-4.f, 5.f, 0.f}).reshape({3, 1});

    const auto [surrogate, log_ratio, ratio] =
        clipped_surrogate({.aim = aim, .rest = rest}, old, aim_advantages, advantages, 0.2f);

    ASSERT_TRUE(torch::allclose(surrogate, aim_advantages + advantages));
    ASSERT_TRUE(torch::allclose(ratio, torch::ones({3, 1})));

    surrogate.sum().backward();
    // at ratio 1, d(surrogate)/d(log-prob of a group) is that group's advantage
    ASSERT_TRUE(torch::allclose(aim.grad(), aim_advantages));
    ASSERT_TRUE(torch::allclose(rest.grad(), advantages));
}

TEST(PpoObjective, GroupsAreClippedApart) {
    const PolicyLogProbs current{
        .aim = torch::full({1, 1}, std::log(2.f)), .rest = torch::full({1, 1}, std::log(0.5f))};
    const PolicyLogProbs old{.aim = torch::zeros({1, 1}), .rest = torch::zeros({1, 1})};

    const auto [surrogate, log_ratio, ratio] =
        clipped_surrogate(current, old, torch::ones({1, 1}), torch::ones({1, 1}), 0.2f);

    // the aim ratio 2 is clipped to 1.2, the rest ratio 0.5 is kept by the min
    ASSERT_NEAR(surrogate.item<float>(), 1.2f + 0.5f, 1e-5f);
    // the whole action did not move: 2 * 0.5
    ASSERT_NEAR(ratio.item<float>(), 1.f, 1e-5f);
}
