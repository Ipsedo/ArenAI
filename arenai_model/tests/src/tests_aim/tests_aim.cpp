#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>

#include <glm/gtc/matrix_transform.hpp>

#include <arenai_model/constants.h>
#include <arenai_model/shell_economy.h>
#include <arenai_model_tests/tests_aim/tests_aim.h>

#include "tank/parts/canon.h"
#include "tank/parts/gun_laying.h"
#include "tank/parts/turret.h"

using namespace arenai;
using namespace arenai::model;
using namespace arenai::controller;

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

    template<typename PartT>
    void steer(PartT &part, const user_input &input) {
        part.apply_input(input);
        part.lay();
    }

    glm::mat3 rotation(const float angle, const glm::vec3 &axis) {
        return glm::mat3(glm::rotate(glm::mat4(1.f), angle, axis));
    }

    float heading(const std::shared_ptr<Item> &item) {
        const glm::vec4 forward = item->get_model_matrix() * glm::vec4(0.f, 0.f, 1.f, 0.f);
        return std::atan2(forward.x, forward.z);
    }

    void
    drive(const Tank &tank, AbstractPhysicEngine &engine, const user_input &input, int frames) {
        for (int i = 0; i < frames; i++) {
            for (const auto &controller: tank.get_controllers()) controller->apply_input(input);
            engine.step(FREQUENCY);
        }
    }

}// namespace

TEST_F(AimTest, FullDeflectionMovesTheTurretByOneStep) {
    const auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto *turret = find_part<TurretItem>(*tank);
    ASSERT_NE(turret, nullptr);

    steer(*turret, aim(1.f, 0.f));

    ASSERT_NEAR(turret->get_angle(), -TURRET_STEP, 1e-6f);
}

TEST_F(AimTest, FullDeflectionMovesTheCanonBySlowerStep) {
    const auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto *canon = find_part<CanonItem>(*tank);
    ASSERT_NE(canon, nullptr);

    steer(*canon, aim(0.f, 1.f));

    ASSERT_NEAR(canon->get_angle(), CANON_STEP, 1e-6f);
    ASSERT_LT(CANON_STEP, TURRET_STEP) << "the canon aims slower than the turret rotates";
}

TEST_F(AimTest, HeldTurretInputKeepsRotating) {
    const auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto *turret = find_part<TurretItem>(*tank);

    for (int i = 0; i < 10; i++) steer(*turret, aim(0.5f, 0.f));

    ASSERT_NEAR(turret->get_angle(), -10.f * 0.5f * TURRET_STEP, 1e-5f);
}

TEST_F(AimTest, CenteredInputLeavesTheTurretStill) {
    const auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto *turret = find_part<TurretItem>(*tank);

    for (int i = 0; i < 10; i++) steer(*turret, aim(1.f, 0.f));
    const float held = turret->get_angle();

    steer(*turret, aim(0.f, 0.f));

    ASSERT_NEAR(turret->get_angle(), held, 1e-6f) << "a centered input must not move the turret";
}

TEST_F(AimTest, TurretWrapsAroundTheBack) {
    const auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto *turret = find_part<TurretItem>(*tank);

    // one full turn minus a step: the next call must land just past the back, not stall
    const int nb_steps = static_cast<int>(2.f * PI / TURRET_STEP);
    for (int i = 0; i < nb_steps; i++) steer(*turret, aim(1.f, 0.f));

    const float before = turret->get_angle();
    steer(*turret, aim(1.f, 0.f));

    ASSERT_LE(std::abs(turret->get_angle()), PI) << "the angle must stay wrapped in [-pi, pi]";
    ASSERT_NEAR(std::remainder(turret->get_angle() - before, 2.f * PI), -TURRET_STEP, 1e-5f);
}

TEST_F(AimTest, CanonStopsAtItsTravelLimit) {
    const auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto *canon = find_part<CanonItem>(*tank);

    for (int i = 0; i < 200; i++) steer(*canon, aim(0.f, 1.f));

    ASSERT_NEAR(canon->get_angle(), CANON_MAX_ANGLE, 1e-6f);

    steer(*canon, aim(0.f, 1.f));
    ASSERT_NEAR(canon->get_angle(), CANON_MAX_ANGLE, 1e-6f);

    steer(*canon, aim(0.f, -1.f));
    ASSERT_NEAR(canon->get_angle(), CANON_MAX_ANGLE - CANON_STEP, 1e-5f)
        << "the canon must come back immediately, not after unwinding";
}

TEST_F(AimTest, PlayerTurretSlewsFasterThanTheEnemy) {
    const auto tank = tank_factory->make_player_tank(file_reader, "player", {0.f, 5.f, 0.f});
    auto *turret = find_part<TurretItem>(*tank);
    ASSERT_NE(turret, nullptr);

    steer(*turret, aim(1.f, 0.f));

    ASSERT_NEAR(turret->get_angle(), -PLAYER_TURRET_STEP, 1e-5f);
    ASSERT_GT(PLAYER_TURRET_STEP, TURRET_STEP);
}

TEST_F(AimTest, StabilisedTurretCounterRotatesTheChassis) {
    const GunLaying gun_laying;

    ASSERT_NEAR(gun_laying.turret_angle(rotation(0.5f, {0.f, 1.f, 0.f})), -0.5f, 1e-5f);
}

TEST_F(AimTest, StabilisedCanonCompensatesTheChassisPitch) {
    const GunLaying gun_laying;

    // nose up by 0.2 rad: the canon lowers by as much to stay level
    ASSERT_NEAR(gun_laying.canon_angle(rotation(-0.2f, {1.f, 0.f, 0.f})), 0.2f, 1e-5f);
}

TEST_F(AimTest, AnchoredSetPointMatchesTheReachedAngles) {
    GunLaying gun_laying;
    const auto chassis_rotation = rotation(0.3f, glm::normalize(glm::vec3(0.2f, 1.f, 0.1f)));
    const auto turret_rotation = chassis_rotation * rotation(0.7f, {0.f, 1.f, 0.f});

    gun_laying.anchor_turret(chassis_rotation, 0.7f);
    gun_laying.anchor_canon(turret_rotation, -0.25f);

    ASSERT_NEAR(gun_laying.turret_angle(chassis_rotation), 0.7f, 1e-4f);
    ASSERT_NEAR(gun_laying.canon_angle(turret_rotation), -0.25f, 1e-4f);
}

TEST_F(AimTest, CanonKeepsItsWorldHeadingWhileTheChassisTurns) {
    add_ground();
    const auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 2.f, 0.f}, GAME_SHELL_ECONOMY);
    drive(*tank, *engine, aim(0.f, 0.f), 120);

    const float chassis_before = heading(tank->get_chassis());
    const float canon_before = heading(tank->get_canon());

    drive(*tank, *engine, {.left_joystick = {1.f, 1.f}, .right_joystick = {0.f, 0.f}}, 180);
    drive(*tank, *engine, aim(0.f, 0.f), 600);

    ASSERT_GT(
        std::abs(std::remainder(heading(tank->get_chassis()) - chassis_before, 2.f * PI)), 0.3f)
        << "the chassis must have turned for the test to mean anything";
    ASSERT_NEAR(std::remainder(heading(tank->get_canon()) - canon_before, 2.f * PI), 0.f, 0.03f);
}

TEST_F(AimTest, CanonKeepsItsWorldHeadingBetweenInputEvents) {
    add_ground();
    const auto tank = tank_factory->make_player_tank(file_reader, "player", {0.f, 2.f, 0.f});
    drive(*tank, *engine, aim(0.f, 0.f), 120);

    const float chassis_before = heading(tank->get_chassis());
    const float canon_before = heading(tank->get_canon());

    // a single event, then the physics runs on its own as for the player
    for (const auto &controller: tank->get_controllers())
        controller->apply_input({.left_joystick = {1.f, 1.f}, .right_joystick = {0.f, 0.f}});
    for (int i = 0; i < 180; i++) engine->step(FREQUENCY);

    ASSERT_GT(
        std::abs(std::remainder(heading(tank->get_chassis()) - chassis_before, 2.f * PI)), 0.3f)
        << "the chassis must have turned for the test to mean anything";
    ASSERT_NEAR(std::remainder(heading(tank->get_canon()) - canon_before, 2.f * PI), 0.f, 0.03f);
}

TEST_F(AimTest, RespawnLaysTheGunWhereTheCanonPoints) {
    add_ground();
    const auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 2.f, 0.f}, GAME_SHELL_ECONOMY);
    auto *turret = find_part<TurretItem>(*tank);
    drive(*tank, *engine, aim(0.f, 0.f), 120);
    drive(*tank, *engine, {.left_joystick = {1.f, 1.f}, .right_joystick = {0.f, 0.f}}, 180);
    drive(*tank, *engine, aim(0.f, 0.f), 600);

    tank->respawn({50.f, 2.f, 50.f});
    const float before = turret->get_angle();
    steer(*turret, aim(0.f, 0.f));

    ASSERT_NEAR(std::remainder(turret->get_angle() - before, 2.f * PI), 0.f, 0.03f)
        << "the reset chassis rotation must not swing the turret";
}

TEST_F(AimTest, TurretFollowsItsSetPointWhileTheChassisTurns) {
    add_ground();
    const auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank", {0.f, 2.f, 0.f}, GAME_SHELL_ECONOMY);
    auto *turret = find_part<TurretItem>(*tank);
    drive(*tank, *engine, aim(0.f, 0.f), 120);

    float max_lag = 0.f;
    for (int i = 0; i < 180; i++) {
        drive(*tank, *engine, {.left_joystick = {1.f, 1.f}, .right_joystick = {0.f, 0.f}}, 1);
        const glm::vec4 forward = glm::inverse(tank->get_chassis()->get_model_matrix())
                                  * turret->get_model_matrix() * glm::vec4(0.f, 0.f, 1.f, 0.f);
        max_lag = std::max(
            max_lag, std::abs(std::remainder(
                         turret->get_angle() - std::atan2(forward.x, forward.z), 2.f * PI)));
    }

    ASSERT_LT(max_lag, glm::radians(1.5f));
}
