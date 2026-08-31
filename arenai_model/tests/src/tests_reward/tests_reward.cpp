//
// Created by samuel on 01/07/2026.
//

#include <cmath>
#include <memory>
#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_model_tests/tests_reward/tests_reward.h>

#include <arenai_model/constants.h>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;
using namespace arenai::controller;

// ========================================================================
// get_reward — base scenarios
// ========================================================================

TEST_F(RewardTest, RewardZeroWhenAliveNoShot) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {20.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    const std::vector tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    const float reward_a = tanks[0]->get_reward();
    const float reward_b = tanks[1]->get_reward();

    // the dense aim shaping leaves a negligible residue when the canon points ~90°
    // away from the enemy, so the reward is near zero rather than exactly zero
    ASSERT_NEAR(reward_a, 0.f, 1e-3f);
    ASSERT_NEAR(reward_b, 0.f, 1e-3f);
}

TEST_F(RewardTest, RewardNegativeWhenDead) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {20.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    const std::vector tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    // damage chassis enough to kill it
    for (const auto chassis_items = tanks[0]->get_items(); const auto &item: chassis_items) {
        if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
            life->receive_damages(
                {.fire_position = glm::vec3(0.f),
                 .impact_position = glm::vec3(0.f),
                 .damages = 1e6f});
            break;
        }
    }

    ASSERT_TRUE(tanks[0]->is_dead());

    const float reward = tanks[0]->get_reward();

    ASSERT_LT(reward, 0.f);
}

TEST_F(RewardTest, DeathPenaltyIsMinusOne) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {20.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    const std::vector tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    // kill by damage → normal death penalty
    for (const auto chassis_items_b = tanks[1]->get_items(); const auto &item: chassis_items_b) {
        if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
            life->receive_damages({.impact_position = glm::vec3(0.f), .damages = 1e6f});
            break;
        }
    }

    const float death_reward = tanks[1]->get_reward();

    // death and suicide share the same penalty so early termination is never an escape;
    // the fatal hit also counts as a received hit (-0.15)
    ASSERT_FLOAT_EQ(death_reward, -1.15f);
}

// ========================================================================
// get_reward — hit/kill rewards via fire
// ========================================================================

TEST_F(RewardTest, RewardPositiveOnHit) {
    add_ground();
    // spawn tanks high enough so all parts start above ground and settle cleanly
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    // settle on ground (300 frames = 5s at 60fps)
    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    // fire from tank_a toward tank_b (canon points +Z by default)
    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    const std::vector tanks{shared_a, shared_b};

    float max_reward = 0.f;
    for (int i = 0; i < 60; i++) {
        engine->step(1.f / 60.f);
        shared_a->tick(tanks);

        max_reward = std::max(shared_a->get_reward(), max_reward);
    }

    ASSERT_TRUE(shared_a->consume_has_hit()) << "shell should have hit the enemy tank";

    ASSERT_FALSE(std::isnan(max_reward)) << "reward should never be NaN";
    ASSERT_FALSE(std::isinf(max_reward)) << "reward should never be Inf";

    ASSERT_GE(max_reward, 0.2f)
        << "reward should be greater than or equal to the hit bonus after hitting an enemy";
}

TEST_F(RewardTest, RewardUnderOneAfterHit) {
    add_ground();
    // spawn tanks high enough so all parts start above ground and settle cleanly
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    // settle on ground (300 frames = 5s at 60fps)
    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    // fire from tank_a toward tank_b (canon points +Z by default)
    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    const std::vector tanks{shared_a, shared_b};

    float max_reward_on_hit = 0.f;
    for (int i = 0; i < 60; i++) {
        engine->step(1.f / 60.f);
        shared_a->tick(tanks);
        max_reward_on_hit = std::max(shared_a->get_reward(), max_reward_on_hit);
    }

    ASSERT_TRUE(shared_a->consume_has_hit()) << "shell should have hit the enemy tank";

    ASSERT_FALSE(std::isnan(max_reward_on_hit)) << "reward should never be NaN";
    ASSERT_FALSE(std::isinf(max_reward_on_hit)) << "reward should never be Inf";

    ASSERT_GE(max_reward_on_hit, 0.2f)
        << "reward should be greater than or equal to the hit bonus after hitting an enemy";

    // no fire, reward under the hit bonus
    constexpr user_input no_fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {false}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(no_fire_input);

    float max_reward_on_no_hit = 0.f;
    for (int i = 0; i < 60; i++) {
        engine->step(1.f / 60.f);
        shared_a->tick(tanks);
        max_reward_on_no_hit = std::max(shared_a->get_reward(), max_reward_on_no_hit);
    }

    ASSERT_FALSE(shared_a->consume_has_hit()) << "no shell should have hit the enemy tank";

    ASSERT_FALSE(std::isnan(max_reward_on_no_hit)) << "reward should never be NaN";
    ASSERT_FALSE(std::isinf(max_reward_on_no_hit)) << "reward should never be Inf";

    ASSERT_LE(max_reward_on_no_hit, 0.2f)
        << "reward should stay under the hit bonus when no shell hit an enemy";
}

// ========================================================================
// get_reward — wrecks are not farmable
// ========================================================================

TEST_F(RewardTest, NoRewardWhenShootingAWreck) {

    constexpr int proprioception_reserve_index = ENEMY_PROPRIOCEPTION_SIZE - 2;

    add_ground();
    // spawn tanks high enough so all parts start above ground and settle cleanly
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    // settle on ground (300 frames = 5s at 60fps)
    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    // kill tank_b by destroying a single part, then close its death as the environment
    // does: the 8 surviving parts must stop paying anything
    for (const auto items_b = shared_b->get_items(); const auto &item: items_b) {
        if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
            life->receive_damages({.impact_position = glm::vec3(0.f), .damages = 1e6f});
            break;
        }
    }
    ASSERT_TRUE(shared_b->is_dead());
    shared_b->on_death();

    const auto shells_ratio_before = shared_a->get_proprioception()[proprioception_reserve_index];

    // fire from tank_a toward the wreck (canon points +Z by default)
    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    const std::vector tanks{shared_a, shared_b};
    for (int i = 0; i < 60; i++) {
        engine->step(1.f / 60.f);
        shared_a->tick(tanks);
    }
    const float reward = shared_a->get_reward();

    ASSERT_FALSE(shared_a->consume_has_hit()) << "a wreck must not count as a hit";
    ASSERT_FLOAT_EQ(reward, 0.f) << "shooting a wreck must pay neither hit nor kill";

    // the shell spent must not be given back: a wreck is not an ammo dump
    ASSERT_LT(shared_a->get_proprioception()[proprioception_reserve_index], shells_ratio_before)
        << "a wreck must not recharge the shell reserve";
}

TEST_F(RewardTest, NoKillRewardWhenHittingAnotherPartOfAWreck) {
    add_ground();
    // spawn tanks high enough so all parts start above ground and settle cleanly
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    // settle on ground (300 frames = 5s at 60fps)
    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    // kill tank_b through a front wheel — on the far side of the wreck, low on the
    // ground, unreachable by the incoming shell. Every other part dies only through
    // kill_life_items() in on_death(): the shell will thus hit a part whose death was
    // never observed by is_already_dead(), the exact spot where a fresh kill (+2)
    // would be wrongly counted if kill() forgot to flag the part as already dead
    for (const auto items_b = shared_b->get_items(); const auto &item: items_b) {
        auto *life = dynamic_cast<LifeItem *>(item.get());
        if (life && item->get_name().find("wheel_right_1") != std::string::npos) {
            life->receive_damages({.impact_position = glm::vec3(0.f), .damages = 1e6f});
            break;
        }
    }
    ASSERT_TRUE(shared_b->is_dead());
    shared_b->on_death();

    // fire from tank_a toward the wreck (canon points +Z by default): the shell hits
    // the rear of the chassis or the turret, never the destroyed front wheel
    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    const std::vector tanks{shared_a, shared_b};
    for (int i = 0; i < 60; i++) {
        engine->step(1.f / 60.f);
        shared_a->tick(tanks);
    }
    const float reward = shared_a->get_reward();

    ASSERT_FALSE(shared_a->consume_has_hit()) << "a wreck part must not count as a hit";
    ASSERT_FALSE(shared_a->consume_has_kill())
        << "a part killed by on_death must not count as a fresh kill";
    ASSERT_FLOAT_EQ(reward, 0.f) << "hitting any part of a wreck must pay neither hit nor kill";
}

// ========================================================================
// get_reward — NaN / Inf stability
// ========================================================================

TEST_F(RewardTest, ZeroRewardWithEmptyTankList) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());

    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->apply_input(fire_input);

    for (int i = 0; i < 60; i++) {
        engine->step(1.f / 60.f);
        shared_tank->tick({});
    }

    // pass an empty tank list — get_nearest_enemy_index returns -1
    const float reward = shared_tank->get_reward();

    ASSERT_FALSE(std::isnan(reward)) << "reward should not be NaN with empty tank list";
    ASSERT_FALSE(std::isinf(reward)) << "reward should not be Inf with empty tank list";

    // firing is free and there is no enemy to sample, so no reward at all
    ASSERT_FLOAT_EQ(reward, 0.f);
}
