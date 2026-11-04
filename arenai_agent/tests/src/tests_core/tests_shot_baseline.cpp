//
// Created by samuel on 04/11/2026.
//

#include <core/shot_baseline.h>
#include <gtest/gtest.h>

using namespace arenai;
using namespace arenai::agent;

namespace {
    constexpr int WINDOW = 8;
}

TEST(ShotBaselineTest, StartsAtZero) {
    const ShotBaseline baseline(WINDOW);

    ASSERT_FLOAT_EQ(baseline.value(), 0.f);
}

TEST(ShotBaselineTest, NoShellLeavesTheValueUnchanged) {
    ShotBaseline baseline(WINDOW);
    baseline.update(1.f, 1);

    baseline.update(100.f, 0);

    ASSERT_FLOAT_EQ(baseline.value(), 1.f);
}

TEST(ShotBaselineTest, ExactMeanUntilTheWindowIsFull) {
    ShotBaseline baseline(WINDOW);

    baseline.update(1.f, 1);
    baseline.update(3.f, 1);
    ASSERT_FLOAT_EQ(baseline.value(), 2.f);

    baseline.update(8.f, 2);
    ASSERT_FLOAT_EQ(baseline.value(), 3.f);
}

TEST(ShotBaselineTest, TracksANewLevelOnceTheWindowIsFull) {
    ShotBaseline baseline(WINDOW);
    for (int i = 0; i < WINDOW; i++) baseline.update(0.f, 1);

    for (int i = 0; i < 8 * WINDOW; i++) baseline.update(5.f, 1);

    ASSERT_NEAR(baseline.value(), 5.f, 1e-2f);
    ASSERT_LT(baseline.value(), 5.f);
}

TEST(ShotBaselineTest, ABatchMovesLikeItsShellsOneByOne) {
    ShotBaseline batched(WINDOW), one_by_one(WINDOW);
    for (int i = 0; i < WINDOW; i++) {
        batched.update(1.f, 1);
        one_by_one.update(1.f, 1);
    }

    batched.update(12.f, 3);
    for (int i = 0; i < 3; i++) one_by_one.update(4.f, 1);

    ASSERT_NEAR(batched.value(), one_by_one.value(), 1e-5f);
}
