//
// Created by claude on 01/07/2026.
//

#include <memory>

#include <arenai_controller/inputs.h>
#include <arenai_model/item.h>
#include <arenai_model_tests/tests_player_tank/tests_player_tank.h>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;
using namespace arenai::controller;

// ========================================================================
// PlayerTank — score tracking
// ========================================================================

TEST_F(PlayerTankTest, ScoreZeroAtCreation) {
    add_ground();
    const auto tank = tank_factory->make_player_tank("player", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    ASSERT_EQ(tank->get_score(), 0);
}

TEST_F(PlayerTankTest, ScoreIncreasesOnHit) {
    add_ground();
    const auto player = tank_factory->make_player_tank("player", {0.f, 5.f, 0.f});
    auto enemy = tank_factory->make_enemy_tank("enemy", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: player->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    ASSERT_GT(player->get_score(), 0) << "score should increase after hitting an enemy";
}

TEST_F(PlayerTankTest, ScoreHigherOnKillThanHit) {
    add_ground();
    const auto player = tank_factory->make_player_tank("player", {0.f, 5.f, 0.f});
    const auto enemy = tank_factory->make_enemy_tank("enemy", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    // weaken the enemy chassis so a single shell kills it
    for (const auto &item: enemy->get_items()) {
        if (auto *life = dynamic_cast<LifeItem *>(item.get())) { life->receive_damages(9.f); }
    }

    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: player->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    // killed_nb * 10 + hit_nb — a kill gives 10 points vs 1 for a hit
    const int score = player->get_score();
    ASSERT_GE(score, 10) << "killing an enemy should give at least 10 points";
}

TEST_F(PlayerTankTest, ScoreDoesNotIncreaseOnSelfHit) {
    add_ground();
    const auto player = tank_factory->make_player_tank("player", {0.f, 5.f, 0.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    // fire without any enemy — shell should hit the ground or expire
    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: player->get_controllers()) ctrl->on_input(fire_input);

    for (int i = 0; i < 1300; i++) engine->step(1.f / 60.f);

    ASSERT_EQ(player->get_score(), 0) << "score should not increase without hitting an enemy";
}

TEST_F(PlayerTankTest, PlayerTankIsDead) {
    add_ground();
    const auto player = tank_factory->make_player_tank("player", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    ASSERT_FALSE(player->is_dead());

    for (const auto &item: player->get_items()) {
        if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
            life->receive_damages(1e6f);
            break;
        }
    }

    ASSERT_TRUE(player->is_dead());
}
