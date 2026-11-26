#include <numbers>

#include <core/panorama_mask_labels.h>
#include <gtest/gtest.h>

using namespace arenai::agent;

namespace {
    constexpr int COLUMNS = 64;
    constexpr float COLUMN_WIDTH = 2.f * std::numbers::pi_v<float> / static_cast<float>(COLUMNS);

    // bearing of the centre of a column
    float centre_of(const int column) {
        return (static_cast<float>(COLUMNS) / 2.f - 0.5f - static_cast<float>(column))
               * COLUMN_WIDTH;
    }

    EnemySighting sighting_at(const int column, const float distance, const bool visible) {
        return {
            .bearing = centre_of(column),
            .angular_half_width = COLUMN_WIDTH / 4.f,
            .distance = distance,
            .visible = visible};
    }
}// namespace

TEST(PanoramaMaskLabelsTest, NoEnemyIsAnEmptyFullyWeightedMask) {
    const auto [labels, weights] = panorama_mask_labels({}, COLUMNS);

    ASSERT_EQ(labels.sizes().vec(), (std::vector<int64_t>{COLUMNS}));
    ASSERT_EQ(labels.sum().item<float>(), 0.f);
    ASSERT_EQ(weights.sum().item<float>(), static_cast<float>(COLUMNS));
}

TEST(PanoramaMaskLabelsTest, VisibleEnemyLabelsItsColumnOnly) {
    const auto [labels, weights] = panorama_mask_labels({sighting_at(20, 80.f, true)}, COLUMNS);

    ASSERT_EQ(labels.sum().item<float>(), 1.f);
    ASSERT_EQ(labels[20].item<float>(), 1.f);
    ASSERT_EQ(weights.sum().item<float>(), static_cast<float>(COLUMNS));
}

TEST(PanoramaMaskLabelsTest, CanonHeadingIsTheMiddleAndLeftIsTheLowIndices) {
    const EnemySighting slightly_left{
        .bearing = COLUMN_WIDTH / 2.f,
        .angular_half_width = COLUMN_WIDTH / 4.f,
        .distance = 50.f,
        .visible = true};

    const auto [labels, weights] = panorama_mask_labels({slightly_left}, COLUMNS);

    ASSERT_EQ(labels[COLUMNS / 2 - 1].item<float>(), 1.f);
}

TEST(PanoramaMaskLabelsTest, OccludedEnemyIsIgnoredNotNegative) {
    const auto [labels, weights] = panorama_mask_labels({sighting_at(20, 80.f, false)}, COLUMNS);

    ASSERT_EQ(labels.sum().item<float>(), 0.f);
    ASSERT_EQ(weights[20].item<float>(), 0.f);
    ASSERT_EQ(weights.sum().item<float>(), static_cast<float>(COLUMNS - 1));
}

TEST(PanoramaMaskLabelsTest, EnemyBeyondTheLabelDistanceIsIgnored) {
    const auto [labels, weights] = panorama_mask_labels({sighting_at(20, 450.f, true)}, COLUMNS);

    ASSERT_EQ(labels.sum().item<float>(), 0.f);
    ASSERT_EQ(weights[20].item<float>(), 0.f);
}

TEST(PanoramaMaskLabelsTest, EnemyOverlappingANeighbourMakesItAmbiguous) {
    // centred in column 20, wide enough to spill over columns 19 and 21
    const EnemySighting wide{
        .bearing = centre_of(20),
        .angular_half_width = COLUMN_WIDTH,
        .distance = 20.f,
        .visible = true};

    const auto [labels, weights] = panorama_mask_labels({wide}, COLUMNS);

    ASSERT_EQ(labels.sum().item<float>(), 1.f);
    ASSERT_EQ(weights[20].item<float>(), 1.f);
    ASSERT_EQ(weights[19].item<float>(), 0.f);
    ASSERT_EQ(weights[21].item<float>(), 0.f);
    ASSERT_EQ(weights.sum().item<float>(), static_cast<float>(COLUMNS - 2));
}

TEST(PanoramaMaskLabelsTest, EnemyStraightBehindWrapsAround) {
    const EnemySighting behind{
        .bearing = std::numbers::pi_v<float>,
        .angular_half_width = COLUMN_WIDTH / 2.f,
        .distance = 20.f,
        .visible = false};

    const auto [labels, weights] = panorama_mask_labels({behind}, COLUMNS);

    ASSERT_EQ(weights[0].item<float>(), 0.f);
    ASSERT_EQ(weights[COLUMNS - 1].item<float>(), 0.f);
}

TEST(PanoramaMaskLabelsTest, VisibleEnemyWinsOverAnOccludedOneInTheSameColumn) {
    const auto [labels, weights] =
        panorama_mask_labels({sighting_at(20, 300.f, false), sighting_at(20, 80.f, true)}, COLUMNS);

    ASSERT_EQ(labels[20].item<float>(), 1.f);
    ASSERT_EQ(weights[20].item<float>(), 1.f);
}
