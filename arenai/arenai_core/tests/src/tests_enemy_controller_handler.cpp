//
// Created by samuel on 01/07/2026.
//

#include <cmath>
#include <memory>

#include <arenai_controller/inputs.h>
#include <arenai_core/enemy_handler.h>
#include <arenai_model/action_stats.h>
#include <arenai_model/constants.h>

#include <../include/arenai_core_tests/tests_enemy_controller_handler.h>

using namespace arenai;
using namespace arenai::core;

// ========================================================================
// Helper: create a handler with given parameters
// ========================================================================

namespace {
    struct HandlerParams {
        float refresh_frequency = 1.f / 60.f;
        float wanted_fire_frequency = 1.f / 6.f;
        float turret_rad_per_second = model::ENEMY_TURRET_RADIAL_VELOCITY;
    };

    std::pair<std::unique_ptr<EnemyControllerHandler>, std::shared_ptr<model::ActionStats>>
    make_handler(const HandlerParams &params = {}) {
        auto stats = std::make_shared<model::ActionStats>();
        auto handler = std::make_unique<EnemyControllerHandler>(
            params.refresh_frequency, params.wanted_fire_frequency, stats,
            params.turret_rad_per_second);
        return {std::move(handler), stats};
    }
}// namespace

// ========================================================================
// Fire throttle — first fire is immediate
// ========================================================================

TEST_F(EnemyControllerHandlerTest, FirstFireIsAllowed) {
    auto [handler, stats] = make_handler();

    constexpr controller::user_input fire_action{{0.f, 0.f}, {0.f, 0.f}, {true}};
    handler->on_event(fire_action);

    ASSERT_TRUE(stats->has_fire());
}

// ========================================================================
// Fire throttle — second consecutive fire is blocked
// ========================================================================

TEST_F(EnemyControllerHandlerTest, SecondConsecutiveFireIsBlocked) {
    auto [handler, stats] = make_handler();

    constexpr controller::user_input fire_action{{0.f, 0.f}, {0.f, 0.f}, {true}};
    handler->on_event(fire_action);
    ASSERT_TRUE(stats->has_fire());

    handler->on_event(fire_action);
    ASSERT_FALSE(stats->has_fire()) << "fire should be throttled on second consecutive frame";
}

// ========================================================================
// Fire throttle — fire allowed after cooldown
// ========================================================================

TEST_F(EnemyControllerHandlerTest, FireAllowedAfterCooldown) {
    constexpr float refresh_freq = 1.f / 60.f;
    constexpr float fire_freq = 1.f / 6.f;
    constexpr int cooldown_frames = static_cast<int>(fire_freq / refresh_freq);

    auto [handler, stats] = make_handler({refresh_freq, fire_freq});

    constexpr controller::user_input fire_action{{0.f, 0.f}, {0.f, 0.f}, {true}};
    constexpr controller::user_input idle_action{{0.f, 0.f}, {0.f, 0.f}, {false}};

    // First fire
    handler->on_event(fire_action);
    ASSERT_TRUE(stats->has_fire());

    // Wait cooldown frames
    for (int i = 0; i < cooldown_frames; i++) handler->on_event(idle_action);

    // Fire again — should be allowed
    handler->on_event(fire_action);
    ASSERT_TRUE(stats->has_fire());
}

// ========================================================================
// Fire throttle — no fire when not pressed
// ========================================================================

TEST_F(EnemyControllerHandlerTest, NoFireWhenNotPressed) {
    auto [handler, stats] = make_handler();

    constexpr controller::user_input idle_action{{1.f, 1.f}, {1.f, 1.f}, {false}};
    handler->on_event(idle_action);

    ASSERT_FALSE(stats->has_fire());
}

// ========================================================================
// Fire throttle — fire before cooldown is blocked
// ========================================================================

TEST_F(EnemyControllerHandlerTest, FireBeforeCooldownIsBlocked) {
    constexpr float refresh_freq = 1.f / 60.f;
    constexpr float fire_freq = 1.f / 6.f;
    constexpr int cooldown_frames = static_cast<int>(fire_freq / refresh_freq);

    auto [handler, stats] = make_handler({refresh_freq, fire_freq});

    constexpr controller::user_input fire_action{{0.f, 0.f}, {0.f, 0.f}, {true}};
    constexpr controller::user_input idle_action{{0.f, 0.f}, {0.f, 0.f}, {false}};

    handler->on_event(fire_action);
    ASSERT_TRUE(stats->has_fire());

    // Wait only half the cooldown
    for (int i = 0; i < cooldown_frames / 2; i++) handler->on_event(idle_action);

    handler->on_event(fire_action);
    ASSERT_FALSE(stats->has_fire()) << "fire should be blocked before cooldown expires";
}

// ========================================================================
// Turret scaling — joystick values are scaled
// ========================================================================

TEST_F(EnemyControllerHandlerTest, TurretScaling) {
    constexpr float refresh_freq = 1.f / 60.f;
    constexpr float turret_rad_per_sec = 2.f;
    constexpr float expected_scale = turret_rad_per_sec * refresh_freq;

    auto stats = std::make_shared<model::ActionStats>();
    const auto handler = std::make_unique<EnemyControllerHandler>(
        refresh_freq, 1.f / 6.f, stats, turret_rad_per_sec);

    constexpr controller::user_input action{{0.f, 0.f}, {1.f, 1.f}, {false}};
    handler->on_event(action);

    const float energy = stats->energy_consumed();
    // energy_consumed = (|lx| + |ly| + |rx_scaled| + |ry_scaled|) / 4
    // with left = (0,0) and right = (1,1) scaled by expected_scale
    constexpr float expected_energy = (0.f + 0.f + expected_scale + expected_scale) / 4.f;
    ASSERT_NEAR(energy, expected_energy, 1e-5f);
}

// ========================================================================
// ActionStats receives processed input
// ========================================================================

TEST_F(EnemyControllerHandlerTest, ActionStatsReceivesInput) {
    auto [handler, stats] = make_handler();

    ASSERT_FLOAT_EQ(stats->energy_consumed(), 0.f);

    constexpr controller::user_input action{{1.f, 0.f}, {0.f, 0.f}, {false}};
    handler->on_event(action);

    ASSERT_GT(stats->energy_consumed(), 0.f);
}

// ========================================================================
// NbFramesToFire computation
// ========================================================================

TEST_F(EnemyControllerHandlerTest, CooldownFrameCount) {
    constexpr float refresh_freq = 1.f / 60.f;
    constexpr float fire_freq = 1.f / 6.f;
    constexpr int expected_frames = static_cast<int>(fire_freq / refresh_freq);

    auto [handler, stats] = make_handler({refresh_freq, fire_freq});

    constexpr controller::user_input fire_action{{0.f, 0.f}, {0.f, 0.f}, {true}};

    // First fire
    handler->on_event(fire_action);
    ASSERT_TRUE(stats->has_fire());

    // Fire should be blocked for exactly (expected_frames - 1) frames
    for (int i = 0; i < expected_frames - 1; i++) {
        handler->on_event(fire_action);
        ASSERT_FALSE(stats->has_fire()) << "fire should be blocked at frame " << (i + 1);
    }

    // Exactly at expected_frames, fire should be allowed again
    handler->on_event(fire_action);
    ASSERT_TRUE(stats->has_fire()) << "fire should be allowed at frame " << expected_frames;
}

// ========================================================================
// Counter saturation — curr_frame does not overflow
// ========================================================================

TEST_F(EnemyControllerHandlerTest, CounterSaturation) {
    auto [handler, stats] = make_handler();

    constexpr controller::user_input idle_action{{0.f, 0.f}, {0.f, 0.f}, {false}};

    // Run many idle frames — way beyond cooldown
    for (int i = 0; i < 10000; i++) handler->on_event(idle_action);

    // Fire should still work after many idle frames
    constexpr controller::user_input fire_action{{0.f, 0.f}, {0.f, 0.f}, {true}};
    handler->on_event(fire_action);
    ASSERT_TRUE(stats->has_fire());
}
