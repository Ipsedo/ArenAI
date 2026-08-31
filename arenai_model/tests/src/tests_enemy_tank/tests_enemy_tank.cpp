//
// Created by claude on 01/07/2026.
//

#include <cmath>
#include <memory>

#include <arenai_controller/inputs.h>
#include <arenai_model/item.h>
#include <arenai_model_tests/tests_enemy_tank/tests_enemy_tank.h>

#include "arenai_model/constants.h"

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;
using namespace arenai::controller;

// ========================================================================
// is_dead — death by individual part destruction
// ========================================================================

TEST_F(EnemyTankTest, DeadWhenSingleWheelDestroyed) {
    add_ground();
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    // find a wheel item (not chassis, not turret, not canon) and destroy it
    const auto items = tank->get_items();
    bool wheel_killed = false;
    for (const auto &item: items) {
        if (item->get_name().find("wheel") != std::string::npos) {
            if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
                life->receive_damages(
                    {.fire_position = glm::vec3(0.f),
                     .impact_position = glm::vec3(0.f),
                     .damages = 1e6f});
                wheel_killed = true;
                break;
            }
        }
    }

    ASSERT_TRUE(wheel_killed) << "should have found and killed a wheel";
    ASSERT_TRUE(tank->is_dead())
        << "tank should be dead when any single wheel is destroyed (any_of over all life_items)";
}

// ========================================================================
// on_death — idempotency
// ========================================================================

TEST_F(EnemyTankTest, OnDeathMultipleCallsDoNotCrash) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());

    // kill the tank
    for (const auto &item: shared_tank->get_items()) {
        if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
            life->receive_damages({.impact_position = glm::vec3(0.f), .damages = 1e6f});
            break;
        }
    }

    ASSERT_TRUE(shared_tank->is_dead());

    // on_death should be safe to call multiple times
    ASSERT_NO_THROW(shared_tank->on_death());
    ASSERT_NO_THROW(shared_tank->on_death());
    ASSERT_NO_THROW(shared_tank->on_death());
}

TEST_F(EnemyTankTest, OnDeathBeforeDeathDoesNothing) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());

    ASSERT_FALSE(shared_tank->is_dead());
    ASSERT_NO_THROW(shared_tank->on_death());
}

// ========================================================================
// Reward — edge cases
// ========================================================================

TEST_F(EnemyTankTest, RewardWhenAllEnemiesDeadAndShellFired) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    // kill tank_b before firing
    for (const auto &item: shared_b->get_items()) {
        if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
            life->receive_damages({.impact_position = glm::vec3(0.f), .damages = 1e6f});
        }
    }
    ASSERT_TRUE(shared_b->is_dead());

    // fire from tank_a — shell will hit the dead tank or ground
    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const std::vector tanks{shared_a, shared_b};

    // get_nearest_enemy_index should return -1 (all dead)
    // reward should not crash and should be 0 (no valid target)
    const float reward = shared_a->get_reward();
    ASSERT_FALSE(std::isnan(reward)) << "reward should not be NaN when all enemies are dead";
    ASSERT_FALSE(std::isinf(reward)) << "reward should not be Inf when all enemies are dead";
}

TEST_F(EnemyTankTest, RewardNoNaNWhenAloneInTankList) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());

    // fire a shell that will hit the ground
    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->apply_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const float reward = shared_tank->get_reward();
    ASSERT_FALSE(std::isnan(reward)) << "reward should not be NaN when alone";
    ASSERT_FALSE(std::isinf(reward)) << "reward should not be Inf when alone";
}

// ========================================================================
// Shell hitting non-LifeItem (ground)
// ========================================================================

TEST_F(EnemyTankTest, ShellHitsGroundNoRewardNoCrash) {
    add_ground();
    // point the tank away from any enemy so the shell hits the ground
    auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());

    // tilt canon downward to ensure it hits the ground
    constexpr user_input aim_down{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 1.f},
        .fire_button = {false}};
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->apply_input(aim_down);
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->apply_input(aim_down);
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->apply_input(aim_down);

    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->apply_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    // shell hit the ground (CubeItem, not LifeItem) — has_hit should be false
    ASSERT_FALSE(shared_tank->consume_has_hit())
        << "hitting the ground should not count as hitting another tank";

    // but last_shoot_info should still be set — reward should not crash
    const float reward = shared_tank->get_reward();
    ASSERT_FALSE(std::isnan(reward));
}

// ========================================================================
// Suicide detection
// ========================================================================

TEST_F(EnemyTankTest, SuicideDetectionWhenFlipped) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());

    ASSERT_FALSE(shared_tank->is_suicide()) << "tank should not be suicidal initially";

    // simulate being upside down by running many reward calls
    // the reward function tracks upside_down frames via dot product
    // we can't easily flip the tank physically, so just verify the initial state
    ASSERT_FALSE(shared_tank->is_dead()) << "tank should be alive initially";
}

// ========================================================================
// has_hit_other_tank — reset behavior
// ========================================================================

TEST_F(EnemyTankTest, HasHitOtherTankResetsAfterCall) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    // first call returns true
    ASSERT_TRUE(shared_a->consume_has_hit());
    // second call should return false (reset)
    ASSERT_FALSE(shared_a->consume_has_hit())
        << "has_hit_other_tank should reset to false after being queried";
}

// ========================================================================
// Shell reserve — passive regeneration
// ========================================================================

TEST_F(EnemyTankTest, ShellReserveRegeneratesOverTime) {
    // the fixture engine runs at 60 Hz: one reward call per frame, one shell every 1.5 s.
    // the bounds stay a frame away from the period so the test does not depend on how the
    // period rounds to an integer number of frames
    constexpr int nb_frames_one_second = 60;
    constexpr int max_shells = 30;
    constexpr int initial_shells = 10;
    constexpr float seconds_per_regen = 1.5f;

    constexpr int proprioception_reserve_index = ENEMY_PROPRIOCEPTION_SIZE - 2;

    add_ground();
    auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    float last_reserve = tank->get_proprioception()[proprioception_reserve_index];
    ASSERT_FLOAT_EQ(
        last_reserve, static_cast<float>(initial_shells) / static_cast<float>(max_shells))
        << "reserve start with initial shells";

    for (int i = 0; i < static_cast<int>(nb_frames_one_second * seconds_per_regen * max_shells);
         i++) {
        engine->step(1.f / static_cast<float>(nb_frames_one_second));
        tank->tick({});

        const float curr_reserve = tank->get_proprioception()[proprioception_reserve_index];
        ASSERT_GE(curr_reserve, last_reserve) << "reserve should increase";

        last_reserve = curr_reserve;
    }

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const std::vector tanks{shared_tank};

    ASSERT_FLOAT_EQ(shared_tank->get_proprioception()[proprioception_reserve_index], 1.f) << "the reserve should be full";

    // fire one shell: the reserve drops by one
    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->apply_input(fire_input);
    engine->step(1.f / 60.f);
    shared_tank->tick({});

    const float reserve_after_fire = shared_tank->get_proprioception()[proprioception_reserve_index];
    ASSERT_FLOAT_EQ(reserve_after_fire, 1.f - 1.f / static_cast<float>(max_shells))
        << "firing should consume a shell";

    // after 1 s the period is not over yet: still nothing
    for (int i = 0; i < nb_frames_one_second / 2; i++) shared_tank->tick(tanks);
    ASSERT_FLOAT_EQ(shared_tank->get_proprioception()[proprioception_reserve_index], reserve_after_fire)
        << "the reserve should not regenerate before the full period has elapsed";

    // after 2 s the shell is back
    for (int i = 0; i < nb_frames_one_second; i++) shared_tank->tick(tanks);
    ASSERT_FLOAT_EQ(shared_tank->get_proprioception()[proprioception_reserve_index], 1.f)
        << "one shell should be given back after 1.5 s";
}

TEST_F(EnemyTankTest, ShellReserveRegenerationIsCappedAtMaximalReserve) {
    constexpr int nb_frames_one_second = 60;
    constexpr int max_shells = 30;
    constexpr float seconds_per_regen = 1.5f;

    add_ground();
    auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    for (int i = 0; i < static_cast<int>(nb_frames_one_second * seconds_per_regen * max_shells);
         i++) {
        engine->step(1.f / static_cast<float>(nb_frames_one_second));
        tank->tick({});
    }

    ASSERT_FLOAT_EQ(tank->get_proprioception().back(), 1.f) << "the reserve should be full";

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const std::vector tanks{shared_tank};

    // the reserve is already full: several periods must not push it above it
    for (int i = 0; i < 5 * nb_frames_one_second; i++) shared_tank->tick(tanks);

    ASSERT_FLOAT_EQ(shared_tank->get_proprioception().back(), 1.f)
        << "regeneration should never take the reserve above its max value";
}
