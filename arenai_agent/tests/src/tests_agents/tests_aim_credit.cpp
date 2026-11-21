#include <agents/aim_credit.h>
#include <gtest/gtest.h>

using namespace arenai::agent;

namespace {
    constexpr float GAMMA = 0.9f;

    // [T, nb_tanks] -> [T, nb_tanks, 1]
    torch::Tensor column(const torch::Tensor &tensor) { return tensor.unsqueeze(-1); }
}// namespace

TEST(AimCreditTest, CreditsTheRiseOfThePotential) {
    const AimPotentials potentials{
        .before = torch::tensor({{0.f}, {0.f}, {1.f}}),
        .after = torch::tensor({{1.f}, {0.f}, {0.f}})};
    const auto no_cut = torch::zeros({3, 1, 1});
    const auto valids = torch::ones({3, 1, 1});

    const auto advantages = aim_potential_advantages(potentials, no_cut, valids, GAMMA);

    ASSERT_EQ(advantages.sizes(), std::vector<int64_t>({3, 1, 1}));
    ASSERT_GT(advantages[0].item<float>(), advantages[1].item<float>());
    ASSERT_GT(advantages[1].item<float>(), advantages[2].item<float>());
    ASSERT_NEAR(advantages.mean().item<float>(), 0.f, 1e-5f);
}

TEST(AimCreditTest, CutStepsCarryNoRise) {
    const AimPotentials potentials{
        .before = torch::tensor({{0.f, 1.f}, {0.f, 1.f}}),
        .after = torch::tensor({{1.f, 0.f}, {1.f, 0.f}})};
    const auto valids = torch::ones({2, 2, 1});
    const auto cuts = column(torch::tensor({{0.f, 0.f}, {1.f, 1.f}}));

    const auto advantages = aim_potential_advantages(potentials, cuts, valids, GAMMA);

    ASSERT_FLOAT_EQ(advantages[1][0].item<float>(), advantages[1][1].item<float>());
    ASSERT_GT(advantages[0][0].item<float>(), advantages[1][0].item<float>());
    ASSERT_LT(advantages[0][1].item<float>(), advantages[1][1].item<float>());
}

TEST(AimCreditTest, DiscountsThePotentialReached) {
    const AimPotentials held{
        .before = torch::tensor({{1.f}, {0.f}}), .after = torch::tensor({{1.f}, {0.f}})};
    const auto no_cut = torch::zeros({2, 1, 1});
    const auto valids = torch::ones({2, 1, 1});

    const auto advantages = aim_potential_advantages(held, no_cut, valids, GAMMA);

    ASSERT_LT(advantages[0].item<float>(), advantages[1].item<float>());
}
