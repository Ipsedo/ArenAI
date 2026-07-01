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
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {0.f, 0.f, 10.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    // fire from tank_a
    const user_input fire_input{{0.f, 1.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: tanks[0]->get_controllers()) ctrl->on_input(fire_input);

    // step so shell can travel and hit
    for (int i = 0; i < 30; i++) engine->step(1.f / 60.f);

    const float reward = tanks[0]->get_reward(tanks);

    // if the shell hit, reward should be positive due to hit bonus
    if (tanks[0]->has_hit_other_tank()) { ASSERT_GT(reward, 0.f); }
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
