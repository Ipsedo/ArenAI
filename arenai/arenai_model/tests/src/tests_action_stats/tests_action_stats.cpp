//
// Created by samuel on 01/07/2026.
//

#include <cmath>

#include <arenai_model/action_stats.h>
#include <arenai_model_tests/tests_action_stats/tests_action_stats.h>

// ========================================================================
// ActionStats
// ========================================================================

TEST_F(ActionStatsTest, InitialStateNoFire) {
    ActionStats stats;

    ASSERT_FALSE(stats.has_fire());
}

TEST_F(ActionStatsTest, InitialEnergyZero) {
    ActionStats stats;

    ASSERT_FLOAT_EQ(stats.energy_consumed(), 0.f);
}

TEST_F(ActionStatsTest, FireButtonPressed) {
    ActionStats stats;

    const user_input input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    stats.process_input(input);

    ASSERT_TRUE(stats.has_fire());
}

TEST_F(ActionStatsTest, FireButtonNotPressed) {
    ActionStats stats;

    const user_input input{{0.f, 0.f}, {0.f, 0.f}, {false}};
    stats.process_input(input);

    ASSERT_FALSE(stats.has_fire());
}

TEST_F(ActionStatsTest, EnergyZeroWhenJoysticksNeutral) {
    ActionStats stats;

    const user_input input{{0.f, 0.f}, {0.f, 0.f}, {false}};
    stats.process_input(input);

    ASSERT_FLOAT_EQ(stats.energy_consumed(), 0.f);
}

TEST_F(ActionStatsTest, EnergyMaxWhenJoysticksFull) {
    ActionStats stats;

    const user_input input{{1.f, 1.f}, {1.f, 1.f}, {false}};
    stats.process_input(input);

    ASSERT_FLOAT_EQ(stats.energy_consumed(), 1.f);
}

TEST_F(ActionStatsTest, EnergyWithNegativeValues) {
    ActionStats stats;

    const user_input input{{-1.f, -1.f}, {-1.f, -1.f}, {false}};
    stats.process_input(input);

    ASSERT_FLOAT_EQ(stats.energy_consumed(), 1.f);
}

TEST_F(ActionStatsTest, EnergyPartialValues) {
    ActionStats stats;

    const user_input input{{0.5f, 0.f}, {0.f, 0.5f}, {false}};
    stats.process_input(input);

    ASSERT_FLOAT_EQ(stats.energy_consumed(), (0.5f + 0.f + 0.f + 0.5f) / 4.f);
}

TEST_F(ActionStatsTest, ProcessInputOverwritesPreviousState) {
    ActionStats stats;

    const user_input input_fire{{0.f, 0.f}, {0.f, 0.f}, {true}};
    stats.process_input(input_fire);
    ASSERT_TRUE(stats.has_fire());

    const user_input input_no_fire{{1.f, 1.f}, {1.f, 1.f}, {false}};
    stats.process_input(input_no_fire);

    ASSERT_FALSE(stats.has_fire());
    ASSERT_FLOAT_EQ(stats.energy_consumed(), 1.f);
}
