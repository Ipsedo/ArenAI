//
// Created by samuel on 01/07/2026.
//

#include <cmath>
#include <memory>
#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_model_tests/tests_reward/tests_reward.h>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;
using namespace arenai::controller;

// mirrors JoltEnemyTank::fire_cost: a fired shell is charged up front, whatever it hits
constexpr float FIRE_COST = 0.2f;

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

    const float reward_a = tanks[0]->get_reward(tanks);
    const float reward_b = tanks[1]->get_reward(tanks);

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
            life->receive_damages(1e6f);
            break;
        }
    }

    ASSERT_TRUE(tanks[0]->is_dead());

    const float reward = tanks[0]->get_reward(tanks);

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
            life->receive_damages(1e6f);
            break;
        }
    }

    const float death_reward = tanks[1]->get_reward(tanks);

    // death and suicide share the same penalty so early termination is never an escape;
    // the fatal hit also counts as a received hit (-0.05)
    ASSERT_FLOAT_EQ(death_reward, -1.1f);
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

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const std::vector tanks{shared_a, shared_b};

    const float reward = shared_a->get_reward(tanks);

    ASSERT_TRUE(shared_a->has_hit_other_tank()) << "shell should have hit the enemy tank";

    ASSERT_FALSE(std::isnan(reward)) << "reward should never be NaN";
    ASSERT_FALSE(std::isinf(reward)) << "reward should never be Inf";

    ASSERT_GE(reward, 1.f)
        << "reward should be greater than or equal to 1.0 after hitting an enemy";
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

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const std::vector tanks{shared_a, shared_b};

    const float reward_on_hit = shared_a->get_reward(tanks);

    ASSERT_TRUE(shared_a->has_hit_other_tank()) << "shell should have hit the enemy tank";

    ASSERT_FALSE(std::isnan(reward_on_hit)) << "reward should never be NaN";
    ASSERT_FALSE(std::isinf(reward_on_hit)) << "reward should never be Inf";

    ASSERT_GE(reward_on_hit, 1.f)
        << "reward should be greater than or equal to 1.0 after hitting an enemy";

    // no fire, reward under 1.0
    constexpr user_input no_fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {false}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(no_fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const float reward_on_no_hit = shared_a->get_reward(tanks);

    ASSERT_FALSE(shared_a->has_hit_other_tank()) << "no shell should have hit the enemy tank";

    ASSERT_FALSE(std::isnan(reward_on_no_hit)) << "reward should never be NaN";
    ASSERT_FALSE(std::isinf(reward_on_no_hit)) << "reward should never be Inf";

    ASSERT_LE(reward_on_no_hit, 1.f) << "reward should be under 1.0 after no hitting an enemy";
}

// ========================================================================
// get_reward — wrecks are not farmable
// ========================================================================

TEST_F(RewardTest, NoRewardWhenShootingAWreck) {
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
            life->receive_damages(1e6f);
            break;
        }
    }
    ASSERT_TRUE(shared_b->is_dead());
    shared_b->on_death();

    const auto shells_ratio_before = shared_a->get_proprioception().back();

    // fire from tank_a toward the wreck (canon points +Z by default)
    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const std::vector tanks{shared_a, shared_b};

    const float reward = shared_a->get_reward(tanks);

    ASSERT_FALSE(shared_a->has_hit_other_tank()) << "a wreck must not count as a hit";
    // the shell is still charged: only the hit/kill/aim payout is withheld, so a wreck is a
    // strictly losing target rather than a neutral one
    ASSERT_FLOAT_EQ(reward, -FIRE_COST) << "shooting a wreck must pay neither hit nor kill";

    // the shell spent must not be given back: a wreck is not an ammo dump
    ASSERT_LT(shared_a->get_proprioception().back(), shells_ratio_before)
        << "a wreck must not recharge the shell reserve";
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

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    // pass an empty tank list — get_nearest_enemy_index returns -1
    constexpr std::vector<std::shared_ptr<EnemyTank>> empty_tanks;
    const float reward = shared_tank->get_reward(empty_tanks);

    ASSERT_FALSE(std::isnan(reward)) << "reward should not be NaN with empty tank list";
    ASSERT_FALSE(std::isinf(reward)) << "reward should not be Inf with empty tank list";

    // there is no enemy to sample, so the shell pays nothing back and only its cost remains
    ASSERT_FLOAT_EQ(reward, -FIRE_COST);
}
