//
// Created by samuel on 05/09/2026.
//

#include <core/spawn_curriculum.h>
#include <gtest/gtest.h>

using namespace arenai;
using namespace arenai::agent;

namespace {
    constexpr float DELTA = 0.05f;
    constexpr float RATIO_LOW = 0.02f;
    constexpr float RATIO_HIGH = 0.04f;
    constexpr int PROBE_WINDOW = 4;
    constexpr std::uint64_t SEED = 42;

    // feed exactly one controller window of probe episodes with the given ratio
    void run_probe_window(SpawnCurriculum &curriculum, const int nb_fires, const int nb_hits) {
        int done = 0;
        while (done < PROBE_WINDOW) {
            curriculum.sample_progress();
            const bool is_probe = curriculum.is_probe();

            curriculum.on_episode_end(nb_fires, nb_hits);
            if (is_probe) done++;
        }
    }
}// namespace

TEST(SpawnCurriculumTest, StartsAtZero) {
    const SpawnCurriculum curriculum(DELTA, RATIO_LOW, RATIO_HIGH, PROBE_WINDOW, 0.2f, SEED);

    ASSERT_FLOAT_EQ(curriculum.upper_bound(), 0.f);
}

TEST(SpawnCurriculumTest, SampleStaysWithinBound) {
    SpawnCurriculum curriculum(DELTA, RATIO_LOW, RATIO_HIGH, PROBE_WINDOW, 0.2f, SEED);

    run_probe_window(curriculum, 100, 10);// grow once so the bound is not 0

    for (int i = 0; i < 1000; i++) {
        const float upper = curriculum.upper_bound();
        const float progress = curriculum.sample_progress();
        ASSERT_GE(progress, 0.f);
        ASSERT_LE(progress, upper);
    }
}

TEST(SpawnCurriculumTest, GrowsAboveHighRatio) {
    SpawnCurriculum curriculum(DELTA, RATIO_LOW, RATIO_HIGH, PROBE_WINDOW, 0.2f, SEED);

    run_probe_window(curriculum, 100, 10);// ratio 0.1 > high

    ASSERT_FLOAT_EQ(curriculum.upper_bound(), DELTA);
}

TEST(SpawnCurriculumTest, ShrinksBelowLowRatio) {
    SpawnCurriculum curriculum(DELTA, RATIO_LOW, RATIO_HIGH, PROBE_WINDOW, 0.2f, SEED);

    run_probe_window(curriculum, 100, 10);
    run_probe_window(curriculum, 100, 10);
    ASSERT_FLOAT_EQ(curriculum.upper_bound(), 2.f * DELTA);

    run_probe_window(curriculum, 100, 1);// ratio 0.01 < low

    ASSERT_FLOAT_EQ(curriculum.upper_bound(), DELTA);
}

TEST(SpawnCurriculumTest, HoldsInsideHysteresisBand) {
    SpawnCurriculum curriculum(DELTA, RATIO_LOW, RATIO_HIGH, PROBE_WINDOW, 0.2f, SEED);

    run_probe_window(curriculum, 100, 10);
    ASSERT_FLOAT_EQ(curriculum.upper_bound(), DELTA);

    run_probe_window(curriculum, 100, 3);// ratio 0.03, between low and high

    ASSERT_FLOAT_EQ(curriculum.upper_bound(), DELTA);
}

TEST(SpawnCurriculumTest, NoFireCountsAsFailure) {
    SpawnCurriculum curriculum(DELTA, RATIO_LOW, RATIO_HIGH, PROBE_WINDOW, 0.2f, SEED);

    run_probe_window(curriculum, 100, 10);
    ASSERT_FLOAT_EQ(curriculum.upper_bound(), DELTA);

    run_probe_window(curriculum, 0, 0);

    ASSERT_FLOAT_EQ(curriculum.upper_bound(), 0.f);
}

TEST(SpawnCurriculumTest, ClampsToUnitInterval) {
    SpawnCurriculum curriculum(DELTA, RATIO_LOW, RATIO_HIGH, PROBE_WINDOW, 0.2f, SEED);

    // more than enough successful windows to reach the top
    for (int i = 0; i < 30; i++) run_probe_window(curriculum, 100, 10);
    ASSERT_FLOAT_EQ(curriculum.upper_bound(), 1.f);

    // and enough failures to reach the bottom
    for (int i = 0; i < 30; i++) run_probe_window(curriculum, 100, 0);
    ASSERT_FLOAT_EQ(curriculum.upper_bound(), 0.f);
}

TEST(SpawnCurriculumTest, NonProbeEpisodesDoNotFeedController) {
    // boundary_proba 0: no episode is a probe, the controller must never move
    // whatever the results
    SpawnCurriculum curriculum(DELTA, RATIO_LOW, RATIO_HIGH, PROBE_WINDOW, 0.f, SEED);
    for (int i = 0; i < 100; i++) {
        curriculum.sample_progress();
        curriculum.on_episode_end(100, 10);
    }

    ASSERT_FLOAT_EQ(curriculum.upper_bound(), 0.f);
}

TEST(SpawnCurriculumTest, ZeroDeltaKeepsBoundFixed) {
    SpawnCurriculum curriculum(0.f, RATIO_LOW, RATIO_HIGH, PROBE_WINDOW, 0.2f, SEED);

    for (int i = 0; i < 10; i++) run_probe_window(curriculum, 100, 10);

    ASSERT_FLOAT_EQ(curriculum.upper_bound(), 0.f);
}
