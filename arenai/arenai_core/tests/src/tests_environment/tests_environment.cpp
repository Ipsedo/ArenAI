//
// Created by samuel on 01/07/2026.
//

#include <cmath>

#include <arenai_controller/inputs.h>
#include <arenai_core_tests/tests_environment/tests_environment.h>

// ========================================================================
// reset_physics — returns correct number of states
// ========================================================================

TEST_F(EnvironmentTest, ResetPhysicsReturnsCorrectNumberOfStates) {
    constexpr int nb_tanks = 3;
    constexpr float frequency = 1.f / 60.f;
    constexpr int vision_h = 16;
    constexpr int vision_w = 16;

    TestTanksEnvironment env(
        file_reader, gl_context, nb_tanks, frequency, vision_h, vision_w, 1, false);

    const auto states = env.reset_physics(100.f, 100.f);

    ASSERT_EQ(static_cast<int>(states.size()), nb_tanks);
}

// ========================================================================
// reset_physics — initial vision is black
// ========================================================================

TEST_F(EnvironmentTest, ResetPhysicsInitialVisionIsBlack) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;
    constexpr int vision_h = 8;
    constexpr int vision_w = 8;

    TestTanksEnvironment env(
        file_reader, gl_context, nb_tanks, frequency, vision_h, vision_w, 1, false);

    for (const auto states = env.reset_physics(100.f, 100.f);
         const auto &[vision, proprioception]: states) {
        ASSERT_EQ(static_cast<int>(vision.pixels.size()), 3 * vision_h * vision_w);
        for (const auto pixel: vision.pixels) { ASSERT_EQ(pixel, 0); }
    }
}

// ========================================================================
// reset_physics — proprioception is non-empty
// ========================================================================

TEST_F(EnvironmentTest, ResetPhysicsProprioceptionNonEmpty) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, gl_context, nb_tanks, frequency, 8, 8, 1, false);

    for (const auto states = env.reset_physics(100.f, 100.f);
         const auto &[vision, proprioception]: states) {
        ASSERT_FALSE(proprioception.empty());
    }
}

// ========================================================================
// reset_physics — on_reset_physics callback is invoked
// ========================================================================

TEST_F(EnvironmentTest, ResetPhysicsCallsOnResetPhysics) {
    constexpr int nb_tanks = 1;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, gl_context, nb_tanks, frequency, 8, 8, 1, false);

    env.reset_physics(100.f, 100.f);

    ASSERT_EQ(env.reset_physics_call_count, 1);
}

// ========================================================================
// step — returns correct number of tuples
// ========================================================================

TEST_F(EnvironmentTest, StepReturnsCorrectNumberOfTuples) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;
    constexpr int vision_h = 16;
    constexpr int vision_w = 16;

    TestTanksEnvironment env(
        file_reader, gl_context, nb_tanks, frequency, vision_h, vision_w, 1, false);

    env.reset_physics(100.f, 100.f);
    env.reset_drawables();

    const std::vector<user_input> actions(nb_tanks, {{0.f, 0.f}, {0.f, 0.f}, {false}});

    const auto results = env.step(frequency, actions);

    ASSERT_EQ(static_cast<int>(results.size()), nb_tanks);

    env.stop_drawing();
}

// ========================================================================
// step — reward and done are valid
// ========================================================================

TEST_F(EnvironmentTest, StepRewardAndDoneAreValid) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, gl_context, nb_tanks, frequency, 16, 16, 1, false);

    env.reset_physics(100.f, 100.f);
    env.reset_drawables();

    const std::vector<user_input> actions(nb_tanks, {{0.f, 0.f}, {0.f, 0.f}, {false}});

    for (const auto results = env.step(frequency, actions);
         const auto &[state, reward, is_done]: results) {
        ASSERT_FALSE(std::isnan(reward)) << "reward should not be NaN";
        ASSERT_FALSE(std::isinf(reward)) << "reward should not be Inf";
    }

    env.stop_drawing();
}

// ========================================================================
// step — on_draw callback is invoked
// ========================================================================

TEST_F(EnvironmentTest, StepCallsOnDraw) {
    constexpr int nb_tanks = 1;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, gl_context, nb_tanks, frequency, 16, 16, 1, false);

    env.reset_physics(100.f, 100.f);
    env.reset_drawables();

    const std::vector<user_input> actions(nb_tanks, {{0.f, 0.f}, {0.f, 0.f}, {false}});

    env.step(frequency, actions);
    ASSERT_EQ(env.draw_call_count, 1);

    env.step(frequency, actions);
    ASSERT_EQ(env.draw_call_count, 2);

    env.stop_drawing();
}

// ========================================================================
// step — multiple steps do not crash
// ========================================================================

TEST_F(EnvironmentTest, MultipleStepsDoNotCrash) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, gl_context, nb_tanks, frequency, 16, 16, 1, false);

    env.reset_physics(100.f, 100.f);
    env.reset_drawables();

    const std::vector<user_input> actions(nb_tanks, {{0.5f, -0.5f}, {0.3f, -0.2f}, {false}});

    for (int i = 0; i < 30; i++) {
        const auto results = env.step(frequency, actions);
        ASSERT_EQ(static_cast<int>(results.size()), nb_tanks);
    }

    env.stop_drawing();
}

// ========================================================================
// reset_drawables — on_reset_drawables callback is invoked
// ========================================================================

TEST_F(EnvironmentTest, ResetDrawablesCallsOnResetDrawables) {
    constexpr int nb_tanks = 1;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, gl_context, nb_tanks, frequency, 16, 16, 1, false);

    env.reset_physics(100.f, 100.f);
    env.reset_drawables();

    ASSERT_EQ(env.reset_drawables_call_count, 1);

    env.stop_drawing();
}

// ========================================================================
// Lifecycle — full cycle without crash
// ========================================================================

TEST_F(EnvironmentTest, FullLifecycle) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, gl_context, nb_tanks, frequency, 16, 16, 1, false);

    // Reset physics
    const auto initial_states = env.reset_physics(100.f, 100.f);
    ASSERT_EQ(static_cast<int>(initial_states.size()), nb_tanks);

    // Start rendering
    env.reset_drawables();

    // Run a few steps
    const std::vector<user_input> actions(nb_tanks, {{0.f, 0.f}, {0.f, 0.f}, {false}});
    for (int i = 0; i < 10; i++) env.step(frequency, actions);

    // Stop rendering
    env.stop_drawing();

    // Reset physics again
    const auto new_states = env.reset_physics(50.f, 50.f);
    ASSERT_EQ(static_cast<int>(new_states.size()), nb_tanks);

    // Restart rendering
    env.reset_drawables();

    // Run more steps
    for (int i = 0; i < 10; i++) env.step(frequency, actions);

    env.stop_drawing();
}

// ========================================================================
// stop_drawing — double call does not crash
// ========================================================================

TEST_F(EnvironmentTest, StopDrawingDoubleCallDoesNotCrash) {
    constexpr int nb_tanks = 1;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, gl_context, nb_tanks, frequency, 16, 16, 1, false);

    env.reset_physics(100.f, 100.f);
    env.reset_drawables();

    env.stop_drawing();
    ASSERT_NO_THROW(env.stop_drawing());
}
