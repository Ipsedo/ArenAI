//
// Created by samuel on 23/09/2026.
//

#include <cmath>
#include <memory>
#include <numbers>

#include <arenai_model/constants.h>
#include <arenai_model_tests/tests_aim/tests_aim.h>

#include "tank/parts/canon.h"
#include "tank/parts/turret.h"

using namespace arenai;
using namespace arenai::model;
using namespace arenai::controller;

// ========================================================================
// Helpers — the parts read right_joystick as an absolute aim target
// ========================================================================

namespace {

    constexpr float PI = std::numbers::pi_v<float>;

    // the fixture builds its engine at 1/60 (see EngineTestFixture::SetUp)
    constexpr float FREQUENCY = 1.f / 60.f;
    constexpr float TURRET_STEP = ENEMY_TURRET_RADIAL_VELOCITY * FREQUENCY;
    constexpr float CANON_STEP = ENEMY_CANON_RADIAL_VELOCITY * FREQUENCY;

    // mirrors the travel limit held by canon.cpp
    constexpr float CANON_MAX_ANGLE = 0.2f * PI;

    template<typename PartT>
    PartT *find_part(const Tank &tank) {
        for (const auto &controller: tank.get_controllers())
            if (auto *part = dynamic_cast<PartT *>(controller.get())) return part;
        return nullptr;
    }

    user_input aim(const float x, const float y) {
        return {.left_joystick = {0.f, 0.f}, .right_joystick = {x, y}};
    }

}// namespace

// ========================================================================
// Absolute aim — a held target settles on an angle instead of drifting
// ========================================================================

TEST_F(AimTest, HeldTurretTargetConvergesAndStops) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *turret = find_part<TurretItem>(*tank);
    ASSERT_NE(turret, nullptr);

    for (int i = 0; i < 200; i++) turret->apply_input(aim(0.5f, 0.f));
    const float settled = turret->get_angle();

    ASSERT_NEAR(settled, 0.5f * PI, 1e-4f);

    turret->apply_input(aim(0.5f, 0.f));
    ASSERT_NEAR(turret->get_angle(), settled, 1e-6f) << "a held target must leave the turret still";
}

TEST_F(AimTest, HeldCanonTargetConvergesAndStops) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *canon = find_part<CanonItem>(*tank);
    ASSERT_NE(canon, nullptr);

    for (int i = 0; i < 200; i++) canon->apply_input(aim(0.f, -0.5f));
    const float settled = canon->get_angle();

    ASSERT_NEAR(settled, -0.5f * CANON_MAX_ANGLE, 1e-4f);

    canon->apply_input(aim(0.f, -0.5f));
    ASSERT_NEAR(canon->get_angle(), settled, 1e-6f);
}

// ========================================================================
// Rate limit — the enemy keeps its own slew speed on each axis
// ========================================================================

TEST_F(AimTest, TurretSlewIsRateLimited) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *turret = find_part<TurretItem>(*tank);

    turret->apply_input(aim(1.f, 0.f));

    ASSERT_NEAR(turret->get_angle(), TURRET_STEP, 1e-6f);
}

TEST_F(AimTest, CanonSlewIsRateLimitedAndSlowerThanTheTurret) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *canon = find_part<CanonItem>(*tank);

    canon->apply_input(aim(0.f, 1.f));

    ASSERT_NEAR(canon->get_angle(), CANON_STEP, 1e-6f);
    ASSERT_LT(CANON_STEP, TURRET_STEP) << "the canon aims slower than the turret rotates";
}

// ========================================================================
// Shortest path — crossing the back stays a small move
// ========================================================================

TEST_F(AimTest, TurretCrossesTheBackInsteadOfUnwinding) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *turret = find_part<TurretItem>(*tank);

    for (int i = 0; i < 200; i++) turret->apply_input(aim(1.f, 0.f));
    const float before = turret->get_angle();
    ASSERT_NEAR(before, PI, 1e-4f);

    // a target 0.02 * PI past the back: a full turn away the other way round
    turret->apply_input(aim(-0.98f, 0.f));

    const float travelled = std::remainder(turret->get_angle() - before, 2.f * PI);
    ASSERT_GT(travelled, 0.f) << "the turret must cross the back, not sweep all the way round";
    ASSERT_NEAR(travelled, TURRET_STEP, 1e-5f);
}

// ========================================================================
// Travel limit — the canon stops at its mechanical range
// ========================================================================

TEST_F(AimTest, CanonStopsAtItsTravelLimit) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *canon = find_part<CanonItem>(*tank);

    for (int i = 0; i < 200; i++) canon->apply_input(aim(0.f, 1.f));

    ASSERT_NEAR(canon->get_angle(), CANON_MAX_ANGLE, 1e-4f);

    canon->apply_input(aim(0.f, 1.f));
    ASSERT_NEAR(canon->get_angle(), CANON_MAX_ANGLE, 1e-6f);
}

// ========================================================================
// The player tank is not rate limited: the mouse keeps its flick aim
// ========================================================================

TEST_F(AimTest, PlayerTurretReachesItsTargetInASingleCall) {
    const auto tank = tank_factory->make_player_tank(file_reader, "player", {0.f, 5.f, 0.f});
    auto *turret = find_part<TurretItem>(*tank);
    auto *canon = find_part<CanonItem>(*tank);
    ASSERT_NE(turret, nullptr);
    ASSERT_NE(canon, nullptr);

    turret->apply_input(aim(0.5f, 1.f));
    canon->apply_input(aim(0.5f, 1.f));

    ASSERT_NEAR(turret->get_angle(), 0.5f * PI, 1e-5f);
    ASSERT_NEAR(canon->get_angle(), CANON_MAX_ANGLE, 1e-5f);
}
