//
// Created by samuel on 01/07/2026.
//

#include <cmath>
#include <fstream>
#include <numeric>

#include <nlohmann/json.hpp>

#include <arenai_controller/inputs.h>

#include <../include/arenai_core_tests/tests_environment.h>

using namespace arenai;
using namespace arenai::core;

// ========================================================================
// reset — returns correct number of states
// ========================================================================

TEST_F(EnvironmentTest, ResetReturnsCorrectNumberOfStates) {
    constexpr int nb_tanks = 3;
    constexpr float frequency = 1.f / 60.f;
    constexpr int vision_h = 16;
    constexpr int vision_w = 16;

    TestTanksEnvironment env(
        file_reader, graphics_backend, nb_tanks, frequency, vision_h, vision_w, 1, false);

    const auto states = env.reset(100.f, 100.f);

    ASSERT_EQ(static_cast<int>(states.size()), nb_tanks);

    env.stop_drawing();
}

// ========================================================================
// reset — initial vision is NOT black
// ========================================================================

TEST_F(EnvironmentTest, ResetInitialVisionIsNotBlack) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;
    constexpr int vision_h = 16;
    constexpr int vision_w = 16;

    TestTanksEnvironment env(
        file_reader, graphics_backend, nb_tanks, frequency, vision_h, vision_w, 1, false);

    for (const auto states = env.reset(100.f, 100.f);
         const auto &[vision, proprioception]: states) {
        ASSERT_EQ(static_cast<int>(vision.pixels.size()), 3 * vision_h * vision_w);

        const int pixel_sum = std::accumulate(
            vision.pixels.begin(), vision.pixels.end(), 0,
            [](const int acc, const uint8_t p) { return acc + static_cast<int>(p); });
        ASSERT_GT(pixel_sum, 0) << "initial vision should not be all black";
    }

    env.stop_drawing();
}

// ========================================================================
// reset — golden image comparison (seeded RNG)
// ========================================================================

TEST_F(EnvironmentTest, ResetGoldenImage) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;
    constexpr int vision_h = 16;
    constexpr int vision_w = 16;

    TestTanksEnvironment env(
        file_reader, graphics_backend, nb_tanks, frequency, vision_h, vision_w, 1, false);

    env.seed(42);

    const auto states = env.reset(100.f, 100.f);

    for (int tank_idx = 0; tank_idx < nb_tanks; tank_idx++) {
        const auto &[vision, proprioception] = states[tank_idx];

        const auto golden_image_path =
            std::filesystem::path(__FILE__).parent_path().parent_path() / "resources"
            / "golden_images" / ("golden_env_reset_tank_" + std::to_string(tank_idx) + ".json");

        if (std::filesystem::exists(golden_image_path)) {
            std::ifstream input_file(golden_image_path);
            nlohmann::json golden_image_json;
            input_file >> golden_image_json;

            const auto golden_pixels = golden_image_json.get<std::vector<uint8_t>>();

            ASSERT_EQ(golden_pixels.size(), vision.pixels.size()) << "tank " << tank_idx;

            // rasterization differs between GPU drivers and llvmpipe versions
            // (headless CI): edge pixels can shift by a full color step, so
            // check mean error and count outliers instead of per-pixel bounds
            constexpr int outlier_threshold = 16;
            double absolute_error_sum = 0.0;
            int nb_outliers = 0;

            for (size_t i = 0; i < golden_pixels.size(); ++i) {
                const int diff = std::abs(golden_pixels[i] - vision.pixels[i]);
                absolute_error_sum += diff;
                if (diff > outlier_threshold) nb_outliers++;
            }

            EXPECT_LE(absolute_error_sum / static_cast<double>(golden_pixels.size()), 2.0)
                << "tank " << tank_idx << " mean absolute error too high";
            EXPECT_LE(nb_outliers, static_cast<int>(golden_pixels.size() / 100))
                << "tank " << tank_idx << " too many pixels differing by more than "
                << outlier_threshold;
        } else {
            std::filesystem::create_directories(golden_image_path.parent_path());
            nlohmann::json output_json(vision.pixels);
            std::ofstream output_file(golden_image_path);
            output_file << output_json;
        }
    }

    env.stop_drawing();
}

// ========================================================================
// reset — proprioception is non-empty
// ========================================================================

TEST_F(EnvironmentTest, ResetProprioceptionNonEmpty) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, graphics_backend, nb_tanks, frequency, 8, 8, 1, false);

    for (const auto states = env.reset(100.f, 100.f);
         const auto &[vision, proprioception]: states) {
        ASSERT_FALSE(proprioception.empty());
    }

    env.stop_drawing();
}

// ========================================================================
// reset — on_reset_physics callback is invoked
// ========================================================================

TEST_F(EnvironmentTest, ResetCallsOnResetPhysics) {
    constexpr int nb_tanks = 1;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, graphics_backend, nb_tanks, frequency, 8, 8, 1, false);

    env.reset(100.f, 100.f);

    ASSERT_EQ(env.reset_physics_call_count, 1);

    env.stop_drawing();
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
        file_reader, graphics_backend, nb_tanks, frequency, vision_h, vision_w, 1, false);

    env.reset(100.f, 100.f);

    const std::vector<controller::user_input> actions(nb_tanks, {{0.f, 0.f}, {0.f, 0.f}, {false}});

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

    TestTanksEnvironment env(file_reader, graphics_backend, nb_tanks, frequency, 16, 16, 1, false);

    env.reset(100.f, 100.f);

    const std::vector<controller::user_input> actions(nb_tanks, {{0.f, 0.f}, {0.f, 0.f}, {false}});

    for (const auto results = env.step(frequency, actions);
         const auto &[state, reward, is_done, is_truncated]: results) {
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

    TestTanksEnvironment env(file_reader, graphics_backend, nb_tanks, frequency, 16, 16, 1, false);

    env.reset(100.f, 100.f);

    const std::vector<controller::user_input> actions(nb_tanks, {{0.f, 0.f}, {0.f, 0.f}, {false}});

    const int draw_count_after_reset = env.draw_call_count;

    env.step(frequency, actions);
    ASSERT_EQ(env.draw_call_count, draw_count_after_reset + 1);

    env.step(frequency, actions);
    ASSERT_EQ(env.draw_call_count, draw_count_after_reset + 2);

    env.stop_drawing();
}

// ========================================================================
// step — multiple steps do not crash
// ========================================================================

TEST_F(EnvironmentTest, MultipleStepsDoNotCrash) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, graphics_backend, nb_tanks, frequency, 16, 16, 1, false);

    env.reset(100.f, 100.f);

    const std::vector<controller::user_input> actions(
        nb_tanks, {{0.5f, -0.5f}, {0.3f, -0.2f}, {false}});

    for (int i = 0; i < 30; i++) {
        const auto results = env.step(frequency, actions);
        ASSERT_EQ(static_cast<int>(results.size()), nb_tanks);
    }

    env.stop_drawing();
}

// ========================================================================
// reset — on_reset_drawables callback is invoked
// ========================================================================

TEST_F(EnvironmentTest, ResetCallsOnResetDrawables) {
    constexpr int nb_tanks = 1;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, graphics_backend, nb_tanks, frequency, 16, 16, 1, false);

    env.reset(100.f, 100.f);

    ASSERT_EQ(env.reset_drawables_call_count, 1);

    env.stop_drawing();
}

// ========================================================================
// Lifecycle — full cycle without crash
// ========================================================================

TEST_F(EnvironmentTest, FullLifecycle) {
    constexpr int nb_tanks = 2;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, graphics_backend, nb_tanks, frequency, 16, 16, 1, false);

    // First episode
    const auto initial_states = env.reset(100.f, 100.f);
    ASSERT_EQ(static_cast<int>(initial_states.size()), nb_tanks);

    const std::vector<controller::user_input> actions(nb_tanks, {{0.f, 0.f}, {0.f, 0.f}, {false}});
    for (int i = 0; i < 10; i++) env.step(frequency, actions);

    // Second episode (reset handles stop_drawing internally)
    const auto new_states = env.reset(50.f, 50.f);
    ASSERT_EQ(static_cast<int>(new_states.size()), nb_tanks);

    for (int i = 0; i < 10; i++) env.step(frequency, actions);

    env.stop_drawing();
}

// ========================================================================
// stop_drawing — double call does not crash
// ========================================================================

TEST_F(EnvironmentTest, StopDrawingDoubleCallDoesNotCrash) {
    constexpr int nb_tanks = 1;
    constexpr float frequency = 1.f / 60.f;

    TestTanksEnvironment env(file_reader, graphics_backend, nb_tanks, frequency, 16, 16, 1, false);

    env.reset(100.f, 100.f);

    env.stop_drawing();
    ASSERT_NO_THROW(env.stop_drawing());
}
