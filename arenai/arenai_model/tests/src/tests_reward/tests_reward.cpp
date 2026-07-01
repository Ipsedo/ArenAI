//
// Created by samuel on 01/07/2026.
//

#include <cmath>
#include <memory>
#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_model/constants.h>
#include <arenai_model_tests/tests_reward/tests_reward.h>

// ========================================================================
// get_reward — base scenarios
// ========================================================================

TEST_F(RewardTest, RewardZeroWhenAliveNoShot) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {20.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    const float reward = tanks[0]->get_reward(tanks);

    ASSERT_FLOAT_EQ(reward, 0.f);
}

TEST_F(RewardTest, RewardNegativeWhenDead) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {20.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    // damage chassis enough to kill it
    auto chassis_items = tanks[0]->get_items();
    for (const auto &item: chassis_items) {
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
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {20.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    // kill by damage → normal death penalty
    auto chassis_items_b = tanks[1]->get_items();
    for (const auto &item: chassis_items_b) {
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
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {0.f, 5.f, 30.f});

    // settle on ground (300 frames = 5s at 60fps)
    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    // fire from tank_a toward tank_b (canon points +Z by default)
    const user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks{shared_a, shared_b};

    ASSERT_TRUE(shared_a->has_hit_other_tank()) << "shell should have hit the enemy tank";

    const float reward = shared_a->get_reward(tanks);
    ASSERT_GT(reward, 0.f) << "reward should be positive after hitting an enemy";
}

// ========================================================================
// get_reward — last_shoot_info reset
// ========================================================================

TEST_F(RewardTest, ShootInfoResetAfterGetReward) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {20.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    tanks[0]->get_reward(tanks);

    // second call without any new shot should give 0 reward (no shoot info)
    const float reward_second = tanks[0]->get_reward(tanks);

    ASSERT_FLOAT_EQ(reward_second, 0.f);
}

// ========================================================================
// get_reward — NaN / Inf stability
// ========================================================================

TEST_F(RewardTest, RewardNoNaNAfterMultipleSteps) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    // fire and let the shell travel
    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: tanks[0]->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const float reward = tanks[0]->get_reward(tanks);

    ASSERT_FALSE(std::isnan(reward)) << "reward should never be NaN";
    ASSERT_FALSE(std::isinf(reward)) << "reward should never be Inf";
}

TEST_F(RewardTest, RewardWithEmptyTankList) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank("tank_a", {0.f, 5.f, 0.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    std::shared_ptr<EnemyTank> shared_tank(tank.release());

    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    // pass an empty tank list — get_nearest_enemy_index returns -1
    std::vector<std::shared_ptr<EnemyTank>> empty_tanks;
    const float reward = shared_tank->get_reward(empty_tanks);

    ASSERT_FALSE(std::isnan(reward)) << "reward should not be NaN with empty tank list";
    ASSERT_FALSE(std::isinf(reward)) << "reward should not be Inf with empty tank list";
}

// ========================================================================
// Reward — accumulation over multiple kills
// ========================================================================

TEST_F(RewardTest, RewardAccumulatesOverMultipleHits) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    std::shared_ptr<EnemyTank> shared_b(tank_b.release());
    std::vector<std::shared_ptr<EnemyTank>> tanks{shared_a, shared_b};

    // first shot
    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const float reward_1 = shared_a->get_reward(tanks);

    // second shot
    constexpr user_input no_fire{{0.f, 0.f}, {0.f, 0.f}, {false}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->on_input(no_fire);
    engine->step(1.f / 60.f);

    for (const auto &ctrl: shared_a->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const float reward_2 = shared_a->get_reward(tanks);

    // both rewards should be individually valid
    ASSERT_FALSE(std::isnan(reward_1));
    ASSERT_FALSE(std::isnan(reward_2));
}
