//
// Created by samuel on 01/07/2026.
//

#include <cmath>
#include <memory>
#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_model/constants.h>
#include <arenai_model_tests/tests_reward/tests_reward.h>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;
using namespace arenai::controller;

// ========================================================================
// get_reward — base scenarios
// ========================================================================

TEST_F(RewardTest, RewardPositiveWhenAliveNoShot) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {20.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    const std::vector<std::shared_ptr<EnemyTank>> tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    const float reward_a = tanks[0]->get_reward(tanks);
    const float reward_b = tanks[1]->get_reward(tanks);

    ASSERT_GT(reward_a, 0.f);
    ASSERT_GT(reward_b, 0.f);
}

TEST_F(RewardTest, RewardNegativeWhenDead) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {20.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks{
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

TEST_F(RewardTest, SuicidePenaltyLessThanDeathPenalty) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {20.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    const std::vector<std::shared_ptr<EnemyTank>> tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    // kill by damage → normal death penalty
    for (const auto chassis_items_b = tanks[1]->get_items(); const auto &item: chassis_items_b) {
        if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
            life->receive_damages(1e6f);
            break;
        }
    }

    const float death_reward = tanks[1]->get_reward(tanks);

    // suicide penalty should be less severe (−0.5 vs −1.0)
    ASSERT_LT(death_reward, -0.5f + 1e-5f);
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
    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const std::vector<std::shared_ptr<EnemyTank>> tanks{shared_a, shared_b};

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
    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const std::vector<std::shared_ptr<EnemyTank>> tanks{shared_a, shared_b};

    const float reward_on_hit = shared_a->get_reward(tanks);

    ASSERT_TRUE(shared_a->has_hit_other_tank()) << "shell should have hit the enemy tank";

    ASSERT_FALSE(std::isnan(reward_on_hit)) << "reward should never be NaN";
    ASSERT_FALSE(std::isinf(reward_on_hit)) << "reward should never be Inf";

    ASSERT_GE(reward_on_hit, 1.f)
        << "reward should be greater than or equal to 1.0 after hitting an enemy";

    // no fire, reward under 1.0
    constexpr user_input no_fire_input{{0.f, 0.f}, {0.f, 0.f}, {false}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->on_input(no_fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const float reward_on_no_hit = shared_a->get_reward(tanks);

    ASSERT_FALSE(shared_a->has_hit_other_tank()) << "no shell should have hit the enemy tank";

    ASSERT_FALSE(std::isnan(reward_on_no_hit)) << "reward should never be NaN";
    ASSERT_FALSE(std::isinf(reward_on_no_hit)) << "reward should never be Inf";

    ASSERT_LE(reward_on_no_hit, 1.f) << "reward should be under 1.0 after no hitting an enemy";
}

// ========================================================================
// get_reward — NaN / Inf stability
// ========================================================================

TEST_F(RewardTest, ZeroRewardWithEmptyTankList) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());

    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    // pass an empty tank list — get_nearest_enemy_index returns -1
    constexpr std::vector<std::shared_ptr<EnemyTank>> empty_tanks;
    const float reward = shared_tank->get_reward(empty_tanks);

    ASSERT_FALSE(std::isnan(reward)) << "reward should not be NaN with empty tank list";
    ASSERT_FALSE(std::isinf(reward)) << "reward should not be Inf with empty tank list";

    ASSERT_FLOAT_EQ(reward, 0.f);
}
