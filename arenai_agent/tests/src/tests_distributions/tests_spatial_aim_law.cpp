#include <cmath>

#include <agents/aim_entropy_alphas.h>
#include <distributions/spatial_aim_law.h>
#include <gtest/gtest.h>
#include <networks/aim_head.h>
#include <torch/torch.h>

using namespace arenai;
using namespace arenai::agent;

namespace {
    constexpr int ROWS = 2, COLS = 4, BATCH = 3;

    SpatialAimLaw make_law(const torch::Tensor &logits, const float mu, const float sigma) {
        return {
            logits, torch::full({BATCH, ROWS * COLS, 2}, mu),
            torch::full({BATCH, ROWS * COLS, 2}, sigma), ROWS, COLS};
    }
}// namespace

TEST(SpatialAimLawTest, UniformCellsHaveLogCellCountEntropy) {
    const auto law = make_law(torch::zeros({BATCH, ROWS * COLS}), 0.f, 0.3f);

    ASSERT_TRUE(torch::allclose(
        law.cell_entropy(), torch::full({BATCH, 1}, std::log(static_cast<float>(ROWS * COLS)))));
}

TEST(SpatialAimLawTest, ModeOfCentredOffsetIsTheCellCentre) {
    // cell 6 = row 1, col 2: x = 2.5 / 4 * 2 - 1 = 0.25, y = 1 - 1.5 / 2 * 2 = -0.5
    auto logits = torch::zeros({BATCH, ROWS * COLS});
    logits.select(1, 6).fill_(10.f);
    const auto law = make_law(logits, 0.f, 0.3f);

    const auto draw = law.mode();
    const auto point = law.point(draw);

    ASSERT_TRUE(torch::all(draw.cell == 6).item<bool>());
    ASSERT_TRUE(torch::allclose(point.select(1, 0), torch::full({BATCH}, 0.25f)));
    ASSERT_TRUE(torch::allclose(point.select(1, 1), torch::full({BATCH}, -0.5f)));
}

TEST(SpatialAimLawTest, OffsetStaysInsideItsCell) {
    const auto law = make_law(torch::zeros({BATCH, ROWS * COLS}), 0.f, 3.f);

    for (int i = 0; i < 20; i++) {
        const auto draw = law.sample();
        const auto point = law.point(draw);
        const auto col = draw.cell.remainder(COLS).to(torch::kFloat);
        const auto centre_x = (col + 0.5f) * (2.f / COLS) - 1.f;
        ASSERT_TRUE(
            torch::all(torch::abs(point.select(1, 0) - centre_x) <= 1.f / COLS).item<bool>());
    }
}

TEST(SpatialAimLawTest, LogProbSplitsIntoCellAndOffset) {
    const auto law = make_law(torch::zeros({BATCH, ROWS * COLS}), 0.f, 0.5f);
    const AimDraw draw{
        .cell = torch::zeros({BATCH}, torch::kInt64), .pre_tanh = torch::zeros({BATCH, 2})};

    // log(1/8) + 2 * (log N(0; 0, 0.5) - log(1 - tanh(0)^2))
    const float expected =
        -std::log(8.f) + 2.f * (-std::log(0.5f) - 0.5f * std::log(2.f * static_cast<float>(M_PI)));
    ASSERT_TRUE(torch::allclose(law.log_prob(draw), torch::full({BATCH, 1}, expected), 1e-4, 1e-5));
}

TEST(SpatialAimLawTest, KlOfALawWithItselfIsZero) {
    const auto law = make_law(torch::randn({BATCH, ROWS * COLS}), 0.2f, 0.4f);

    ASSERT_TRUE(torch::allclose(law.kl(law), torch::zeros({BATCH, 1}), 1e-5, 1e-6));
}

TEST(SpatialAimLawTest, AcceptsAnyLeadingDimensions) {
    const SpatialAimLaw law(
        torch::zeros({2, 5, ROWS * COLS}), torch::zeros({2, 5, ROWS * COLS, 2}),
        torch::full({2, 5, ROWS * COLS, 2}, 0.3f), ROWS, COLS);

    const auto draw = law.sample();

    ASSERT_EQ(draw.cell.sizes(), (std::vector<int64_t>{2, 5}));
    ASSERT_EQ(law.point(draw).sizes(), (std::vector<int64_t>{2, 5, 2}));
    ASSERT_EQ(law.log_prob(draw).sizes(), (std::vector<int64_t>{2, 5, 1}));
    ASSERT_EQ(law.offset_entropy().sizes(), (std::vector<int64_t>{2, 5, 2}));
}

TEST(AimHeadTest, StartsUniformAtTheInitialSigma) {
    AimHead head(8, 5, 0.3f);

    const auto law = head.forward(torch::randn({BATCH, 8, ROWS, COLS}), torch::randn({BATCH, 5}));

    // near-zero weights: about log(R * C) whatever the inputs
    ASSERT_TRUE(torch::allclose(
        law.cell_entropy(), torch::full({BATCH, 1}, std::log(static_cast<float>(ROWS * COLS))),
        1e-2, 1e-2));
    const auto draw = law.mode();
    ASSERT_TRUE(torch::all(torch::abs(law.point(draw)) <= 1.f).item<bool>());
}

TEST(AimEntropyAlphasTest, EntropyUnderTargetRaisesAlpha) {
    const AimEntropyAlphas alphas(1e-2f, 1e-2f, 0.69f, std::vector{-1.f, -1.f});

    const auto cell_before = alphas.cell().detach().clone();
    const auto offset_before = alphas.offset().detach().clone();

    alphas.zero_grad();
    alphas.backward(torch::full({8, 1}, 0.1f), torch::full({8, 2}, -2.f), 1.f);
    alphas.step();

    ASSERT_TRUE(torch::all(alphas.cell() > cell_before).item<bool>());
    ASSERT_TRUE(torch::all(alphas.offset() > offset_before).item<bool>());
    ASSERT_EQ(alphas.get_metrics().size(), 3);
}

TEST(AimEntropyAlphasTest, RejectsAWrongOffsetTargetSize) {
    ASSERT_THROW(AimEntropyAlphas(1e-3f, 1e-3f, 0.69f, std::vector{-1.f}), std::invalid_argument);
}
