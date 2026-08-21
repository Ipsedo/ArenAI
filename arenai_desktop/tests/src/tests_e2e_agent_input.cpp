//
// Created by samuel on 28/07/2026.
//

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>

#include <core/game_environment.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <arenai_desktop_tests/headless_windowed_backend.h>
#include <arenai_desktop_tests/interceptor_agent.h>

using namespace arenai;
using namespace arenai::desktop;

// ========================================================================
// End-to-end: DesktopGameEnvironment -> AbstractAgent::act()
// Mirrors run_game()'s loop with the interceptor in the SAC agent's seat;
// the player view goes to a headless no-op backend, the enemy visions are
// rendered by the environment's real offscreen backend. The player tank is
// part of the scene, hence desktop-specific golden images.
// ========================================================================

namespace {

    constexpr int NB_TANKS = 2;
    constexpr int VISION_HEIGHT = 16;
    constexpr int VISION_WIDTH = 16;
    constexpr float FREQUENCY = 1.f / 60.f;
    constexpr int NB_STEPS = 30;

    std::unique_ptr<DesktopGameEnvironment> make_environment() {
        auto env = std::make_unique<DesktopGameEnvironment>(
            ARENAI_ASSETS_DIR, std::make_shared<HeadlessWindowedBackend>(), NB_TANKS, VISION_HEIGHT,
            VISION_WIDTH, FREQUENCY, ControllerKind::Gamepad);
        env->seed(42);
        return env;
    }

    // the run_game() loop shape: act on the current states, step the
    // environment with the returned actions, feed the next states back
    void run_act_step_loop(
        core::BaseTanksEnvironment &env, InterceptorAgent &agent, std::vector<core::State> states) {
        auto actions = agent.act(states, VISION_HEIGHT, VISION_WIDTH);

        for (int i = 0; i < NB_STEPS; i++) {
            const auto steps = env.step(FREQUENCY, actions);

            states.clear();
            for (const auto &[state, reward, done]: steps) states.push_back(state);

            actions = agent.act(states, VISION_HEIGHT, VISION_WIDTH);
        }
    }

    // same tolerance as arenai_core's ResetGoldenImage: rasterization shifts
    // between llvmpipe versions, so mean error + outlier count, not per-pixel
    void expect_matches_golden(
        const std::filesystem::path &golden_path, const std::vector<uint8_t> &pixels,
        const std::string &label) {
        std::ifstream input_file(golden_path);
        nlohmann::json golden_json;
        input_file >> golden_json;

        const auto golden_pixels = golden_json.get<std::vector<uint8_t>>();

        ASSERT_EQ(golden_pixels.size(), pixels.size()) << label;

        constexpr int outlier_threshold = 16;
        double absolute_error_sum = 0.0;
        int nb_outliers = 0;

        for (size_t i = 0; i < golden_pixels.size(); ++i) {
            const int diff = std::abs(golden_pixels[i] - pixels[i]);
            absolute_error_sum += diff;
            if (diff > outlier_threshold) nb_outliers++;
        }

        EXPECT_LE(absolute_error_sum / static_cast<double>(golden_pixels.size()), 2.0)
            << label << " mean absolute error too high";
        EXPECT_LE(nb_outliers, static_cast<int>(golden_pixels.size() / 100))
            << label << " too many pixels differing by more than " << outlier_threshold;
    }

    void
    write_golden(const std::filesystem::path &golden_path, const std::vector<uint8_t> &pixels) {
        std::filesystem::create_directories(golden_path.parent_path());
        const nlohmann::json output_json(pixels);
        std::ofstream output_file(golden_path);
        output_file << output_json;
    }

    std::filesystem::path golden_image_path(const std::string &name) {
        // arenai_desktop/tests/src/ -> arenai_desktop/tests/
        return std::filesystem::path(__FILE__).parent_path().parent_path() / "resources"
               / "golden_images" / (name + ".json");
    }

}// namespace

// ========================================================================
// act() receives structurally valid states on every loop iteration
// ========================================================================

TEST(DesktopAgentInputEndToEnd, AgentReceivesValidStates) {
    const auto env = make_environment();
    InterceptorAgent agent;

    run_act_step_loop(*env, agent, env->reset(100.f, 100.f));

    ASSERT_EQ(static_cast<int>(agent.received_states.size()), NB_STEPS + 1);
    ASSERT_EQ(agent.last_vision_height, VISION_HEIGHT);
    ASSERT_EQ(agent.last_vision_width, VISION_WIDTH);

    for (const auto &states: agent.received_states) {
        ASSERT_EQ(static_cast<int>(states.size()), NB_TANKS);

        for (const auto &[vision, proprioception]: states) {
            ASSERT_EQ(static_cast<int>(vision.pixels.size()), 3 * VISION_HEIGHT * VISION_WIDTH);

            const int pixel_sum = std::accumulate(
                vision.pixels.begin(), vision.pixels.end(), 0,
                [](const int acc, const uint8_t p) { return acc + static_cast<int>(p); });
            ASSERT_GT(pixel_sum, 0) << "vision handed to the agent should not be all black";

            ASSERT_FALSE(proprioception.empty());
            for (const float value: proprioception) ASSERT_TRUE(std::isfinite(value));
        }
    }

    env->stop_drawing();
}

// ========================================================================
// what reaches act() matches the golden images (seeded RNG, pinned env)
// ========================================================================

TEST(DesktopAgentInputEndToEnd, AgentInputMatchesGoldenImages) {
#ifndef ARENAI_REGENERATE_GOLDEN_IMAGES
    // see arenai_core's ResetGoldenImage: only the pinned environment
    // (scripts/goldens_docker.sh, also used by the CI) can compare goldens
    if (std::getenv("ARENAI_PINNED_RENDER_ENV") == nullptr)
        GTEST_SKIP() << "golden comparison needs the pinned render environment: "
                        "run ./scripts/goldens_docker.sh";
#endif

    const auto env = make_environment();
    InterceptorAgent agent;

    run_act_step_loop(*env, agent, env->reset(100.f, 100.f));

    for (int tank_idx = 0; tank_idx < NB_TANKS; tank_idx++) {
        const std::vector<std::pair<std::filesystem::path, const std::vector<uint8_t> *>> goldens =
            {
                {golden_image_path("golden_desktop_reset_tank_" + std::to_string(tank_idx)),
                 &agent.received_states.front()[tank_idx].vision.pixels},
                {golden_image_path(
                     "golden_desktop_step" + std::to_string(NB_STEPS) + "_tank_"
                     + std::to_string(tank_idx)),
                 &agent.received_states.back()[tank_idx].vision.pixels},
            };

        for (const auto &[golden_path, pixels]: goldens) {
#ifdef ARENAI_REGENERATE_GOLDEN_IMAGES
            // rebuild mode: always overwrite the golden below
            if (false) {
#else
            if (std::filesystem::exists(golden_path)) {
#endif
                expect_matches_golden(golden_path, *pixels, golden_path.stem().string());
            } else {
                write_golden(golden_path, *pixels);
            }
        }
    }

    env->stop_drawing();
}
