//
// Created by claude on 08/11/2026.
//

#include <cmath>

#include <agents/sac/sac_discrete.h>
#include <distributions/bernoulli.h>
#include <gtest/gtest.h>

using namespace arenai;
using namespace arenai::agent;

TEST(SacDiscrete, CombinationsEnumerateEveryBinaryValue) {
    const auto combinations = discrete_combinations(2, torch::TensorOptions().dtype(torch::kFloat));

    ASSERT_EQ(combinations.sizes(), (std::vector<int64_t>{4, 2}));
    // action j of combination k is bit j of k
    const auto expected = torch::tensor({{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}});
    ASSERT_TRUE(torch::equal(combinations, expected));
}

TEST(SacDiscrete, CombinationIndexInvertsCombinations) {
    for (const int64_t nb_actions: {1, 2, 3}) {
        const auto combinations =
            discrete_combinations(nb_actions, torch::TensorOptions().dtype(torch::kFloat));
        const auto expected = torch::arange(int64_t{1} << nb_actions, torch::kLong);

        ASSERT_TRUE(torch::equal(combination_index(combinations), expected));
    }
}

TEST(SacDiscrete, JointProbabilitiesSumToOne) {
    const auto probas = torch::rand({5, 3});
    const auto joint = discrete_joint(probas, torch::ones({5, 3}));

    ASSERT_EQ(joint.probas.sizes(), (std::vector<int64_t>{5, 8}));
    ASSERT_TRUE(torch::allclose(joint.probas.sum(-1), torch::ones({5}), 1e-5, 1e-5));
    // unmasked: the log-probabilities are the log of the probabilities
    ASSERT_TRUE(torch::allclose(joint.log_probas.exp(), joint.probas, 1e-5, 1e-5));
}

TEST(SacDiscrete, JointMatchesIndependentBernoullis) {
    const auto probas = torch::tensor({{0.2f, 0.7f}});
    const auto joint = discrete_joint(probas, torch::ones({1, 2}));

    // k = 1 : action 0 engaged, action 1 not
    ASSERT_NEAR(joint.probas[0][1].item<float>(), 0.2f * 0.3f, 1e-6f);
    // k = 3 : both engaged
    ASSERT_NEAR(joint.probas[0][3].item<float>(), 0.2f * 0.7f, 1e-6f);
}

TEST(SacDiscrete, MaskedActionIsNeverEngagedAndCarriesNoEntropy) {
    const auto probas = torch::tensor({{0.6f, 0.3f}});
    // action 0 (fire) without effect
    const auto joint = discrete_joint(probas, torch::tensor({{0.f, 1.f}}));

    // combinations engaging the fire get no weight
    ASSERT_LT(joint.probas[0][1].item<float>(), 1e-6f);
    ASSERT_LT(joint.probas[0][3].item<float>(), 1e-6f);
    ASSERT_NEAR(joint.probas[0][0].item<float>(), 0.7f, 1e-5f);
    ASSERT_NEAR(joint.probas[0][2].item<float>(), 0.3f, 1e-5f);

    // the joint entropy is the one of the remaining action alone
    const auto expected = bernoulli_entropy(torch::tensor({{0.3f}})).item<float>();
    ASSERT_NEAR(discrete_joint_entropy(joint).item<float>(), expected, 1e-4f);
}

TEST(SacDiscrete, ActionEntropiesAreTheMarginalsAndSumToTheJoint) {
    const auto probas = torch::tensor({{0.2f, 0.7f}, {0.6f, 0.3f}});
    const auto mask = torch::tensor({{1.f, 1.f}, {0.f, 1.f}});
    const auto joint = discrete_joint(probas, mask);

    const auto entropies = discrete_action_entropies(joint);
    ASSERT_EQ(entropies.sizes(), (std::vector<int64_t>{2, 2}));

    // each action's entropy is its own Bernoulli's, a masked action carries none
    ASSERT_TRUE(torch::allclose(entropies, bernoulli_entropy(probas) * mask, 1e-5, 1e-5));
    ASSERT_TRUE(
        torch::allclose(entropies.sum(-1, true), discrete_joint_entropy(joint), 1e-5, 1e-5));
}

TEST(SacDiscrete, UniformJointHasMaximumEntropy) {
    const auto joint = discrete_joint(torch::full({1, 2}, 0.5f), torch::ones({1, 2}));

    ASSERT_NEAR(discrete_joint_entropy(joint).item<float>(), 2.f * std::log(2.f), 1e-5f);
}
