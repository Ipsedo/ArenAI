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
// Helpers — the parts read right_joystick as an aim rate in [-1, 1]
// ========================================================================

namespace {

    constexpr float PI = std::numbers::pi_v<float>;

    // the fixture builds its engine at 1/60 (see EngineTestFixture::SetUp)
    constexpr float FREQUENCY = 1.f / 60.f;
    constexpr float TURRET_STEP = ENEMY_TURRET_RADIAL_VELOCITY * FREQUENCY;
    constexpr float CANON_STEP = ENEMY_CANON_RADIAL_VELOCITY * FREQUENCY;
    constexpr float PLAYER_TURRET_STEP = PLAYER_TURRET_RADIAL_VELOCITY * FREQUENCY;

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
// Rate — one call moves the aim by one step, a held input keeps moving it
// ========================================================================

TEST_F(AimTest, FullDeflectionMovesTheTurretByOneStep) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *turret = find_part<TurretItem>(*tank);
    ASSERT_NE(turret, nullptr);

    turret->apply_input(aim(1.f, 0.f));

    ASSERT_NEAR(turret->get_angle(), -TURRET_STEP, 1e-6f);
}

TEST_F(AimTest, FullDeflectionMovesTheCanonBySlowerStep) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *canon = find_part<CanonItem>(*tank);
    ASSERT_NE(canon, nullptr);

    canon->apply_input(aim(0.f, 1.f));

    ASSERT_NEAR(canon->get_angle(), CANON_STEP, 1e-6f);
    ASSERT_LT(CANON_STEP, TURRET_STEP) << "the canon aims slower than the turret rotates";
}

TEST_F(AimTest, HeldTurretInputKeepsRotating) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *turret = find_part<TurretItem>(*tank);

    for (int i = 0; i < 10; i++) turret->apply_input(aim(0.5f, 0.f));

    ASSERT_NEAR(turret->get_angle(), -10.f * 0.5f * TURRET_STEP, 1e-5f);
}

TEST_F(AimTest, CenteredInputLeavesTheTurretStill) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *turret = find_part<TurretItem>(*tank);

    for (int i = 0; i < 10; i++) turret->apply_input(aim(1.f, 0.f));
    const float held = turret->get_angle();

    turret->apply_input(aim(0.f, 0.f));

    ASSERT_NEAR(turret->get_angle(), held, 1e-6f) << "a centered input must not move the turret";
}

// ========================================================================
// The turret rotates freely: crossing the back is just another step
// ========================================================================

TEST_F(AimTest, TurretWrapsAroundTheBack) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *turret = find_part<TurretItem>(*tank);

    // one full turn minus a step: the next call must land just past the back, not stall
    const int nb_steps = static_cast<int>(2.f * PI / TURRET_STEP);
    for (int i = 0; i < nb_steps; i++) turret->apply_input(aim(1.f, 0.f));

    const float before = turret->get_angle();
    turret->apply_input(aim(1.f, 0.f));

    ASSERT_LE(std::abs(turret->get_angle()), PI) << "the angle must stay wrapped in [-pi, pi]";
    ASSERT_NEAR(std::remainder(turret->get_angle() - before, 2.f * PI), -TURRET_STEP, 1e-5f);
}

// ========================================================================
// Travel limit — the canon stops at its mechanical range, without windup
// ========================================================================

TEST_F(AimTest, CanonStopsAtItsTravelLimit) {
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, false);
    auto *canon = find_part<CanonItem>(*tank);

    for (int i = 0; i < 200; i++) canon->apply_input(aim(0.f, 1.f));

    ASSERT_NEAR(canon->get_angle(), CANON_MAX_ANGLE, 1e-6f);

    canon->apply_input(aim(0.f, 1.f));
    ASSERT_NEAR(canon->get_angle(), CANON_MAX_ANGLE, 1e-6f);

    canon->apply_input(aim(0.f, -1.f));
    ASSERT_NEAR(canon->get_angle(), CANON_MAX_ANGLE - CANON_STEP, 1e-5f)
        << "the canon must come back immediately, not after unwinding";
}

// ========================================================================
// The player aims far faster: the mouse sets its slew speed
// ========================================================================

TEST_F(AimTest, PlayerTurretSlewsFasterThanTheEnemy) {
    const auto tank = tank_factory->make_player_tank(file_reader, "player", {0.f, 5.f, 0.f});
    auto *turret = find_part<TurretItem>(*tank);
    ASSERT_NE(turret, nullptr);

    turret->apply_input(aim(1.f, 0.f));

    ASSERT_NEAR(turret->get_angle(), -PLAYER_TURRET_STEP, 1e-5f);
    ASSERT_GT(PLAYER_TURRET_STEP, TURRET_STEP);
}
