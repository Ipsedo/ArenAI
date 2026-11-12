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

    const auto [surrogate, log_ratio, ratio, kls] =
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

    const auto [surrogate, log_ratio, ratio, kls] =
        clipped_surrogate(current, old, torch::ones({1, 1}), torch::ones({1, 1}), 0.2f);

    // the aim ratio 2 is clipped to 1.2, the rest ratio 0.5 is kept by the min
    ASSERT_NEAR(surrogate.item<float>(), 1.2f + 0.5f, 1e-5f);
    // the whole action did not move: 2 * 0.5
    ASSERT_NEAR(ratio.item<float>(), 1.f, 1e-5f);
}

TEST(PpoObjective, AStoppedGroupLeavesTheSurrogate) {
    const auto aim = torch::zeros({3, 1}, torch::requires_grad());
    const auto rest = torch::zeros({3, 1}, torch::requires_grad());
    const PolicyLogProbs old{.aim = torch::zeros({3, 1}), .rest = torch::zeros({3, 1})};

    const auto aim_advantages = torch::tensor({1.f, -2.f, 3.f}).reshape({3, 1});
    const auto advantages = torch::tensor({-4.f, 5.f, 0.f}).reshape({3, 1});

    const auto [surrogate, log_ratio, ratio, kls] = clipped_surrogate(
        {.aim = aim, .rest = rest}, old, aim_advantages, advantages, 0.2f,
        {.aim = false, .rest = true});

    ASSERT_TRUE(torch::allclose(surrogate, advantages));

    surrogate.sum().backward();
    ASSERT_FALSE(aim.grad().defined() && aim.grad().abs().sum().item<float>() > 0.f);
    ASSERT_TRUE(torch::allclose(rest.grad(), advantages));
}

TEST(PpoObjective, EachGroupHasItsOwnKl) {
    const PolicyLogProbs current{
        .aim = torch::full({1, 1}, std::log(2.f)), .rest = torch::full({1, 1}, std::log(0.5f))};
    const PolicyLogProbs old{.aim = torch::zeros({1, 1}), .rest = torch::zeros({1, 1})};

    const auto [surrogate, log_ratio, ratio, kls] =
        clipped_surrogate(current, old, torch::ones({1, 1}), torch::ones({1, 1}), 0.2f);

    // r - 1 - log r
    ASSERT_NEAR(kls.aim.item<float>(), 1.f - std::log(2.f), 1e-5f);
    ASSERT_NEAR(kls.rest.item<float>(), -0.5f + std::log(2.f), 1e-5f);
    ASSERT_NEAR(kls.joint.item<float>(), 0.f, 1e-5f);
}

TEST(PpoObjective, ContinuousGroupMaskKeepsTheActiveActions) {
    // dir_x, dir_y, canon_x, canon_y
    const auto aim_only = continuous_group_mask({.aim = true, .rest = false}, 4, torch::kCPU);
    const auto rest_only = continuous_group_mask({.aim = false, .rest = true}, 4, torch::kCPU);

    ASSERT_TRUE(torch::equal(aim_only, torch::tensor({0.f, 0.f, 1.f, 1.f})));
    ASSERT_TRUE(torch::equal(rest_only, torch::tensor({1.f, 1.f, 0.f, 0.f})));
}

TEST(PpoObjective, GroupKlStopStopsEachGroupApart) {
    GroupKlStop kl_stop(0.05f);
    kl_stop.on_planned_steps(4);

    ASSERT_FALSE(kl_stop.stop_exceeded({.joint = 0.07f, .aim = 0.07f, .rest = 0.f}));
    kl_stop.on_step();

    // past 1.5 * 0.05 for the aim group only
    ASSERT_TRUE(kl_stop.stop_exceeded({.joint = 0.08f, .aim = 0.08f, .rest = 0.f}));
    ASSERT_FALSE(kl_stop.active().aim);
    ASSERT_TRUE(kl_stop.active().rest);
    kl_stop.on_step();
    kl_stop.on_step();

    ASSERT_TRUE(kl_stop.stop_exceeded({.joint = 0.1f, .aim = 0.f, .rest = 0.1f}));
    ASSERT_FALSE(kl_stop.any_active());

    ASSERT_FLOAT_EQ(kl_stop.aim_skip_fraction(), 1.f - 1.f / 4.f);
    ASSERT_FLOAT_EQ(kl_stop.rest_skip_fraction(), 1.f - 3.f / 4.f);
}

TEST(PpoObjective, GroupKlStopNeverStopsWithoutTarget) {
    GroupKlStop kl_stop(0.f);

    ASSERT_FALSE(kl_stop.stop_exceeded({.joint = 100.f, .aim = 100.f, .rest = 100.f}));
    ASSERT_TRUE(kl_stop.any_active());
    ASSERT_FLOAT_EQ(kl_stop.aim_skip_fraction(), 0.f);
}
