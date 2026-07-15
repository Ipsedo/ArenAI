//
// Created by samuel on 01/07/2026.
//

#include <cmath>
#include <memory>

#include <arenai_controller/inputs.h>
#include <arenai_model/action_stats.h>
#include <arenai_model/constants.h>
#include <arenai_model_tests/tests_shell/tests_shell.h>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;
using namespace arenai::controller;

// ========================================================================
// ShellItem — via canon fire
// ========================================================================

TEST_F(ShellTest, FireCreatesShellItem) {
    add_ground();
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const int count_before = static_cast<int>(engine->get_items().size());

    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: tank->get_controllers()) ctrl->apply_input(fire_input);

    engine->step(1.f / 60.f);

    const int count_after = static_cast<int>(engine->get_items().size());

    ASSERT_GT(count_after, count_before) << "firing should create a new shell item";
}

TEST_F(ShellTest, ShellDestroyedAfterLifetime) {
    add_ground();
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const int count_before_fire = static_cast<int>(engine->get_items().size());

    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: tank->get_controllers()) ctrl->apply_input(fire_input);

    engine->step(1.f / 60.f);

    const int count_with_shell = static_cast<int>(engine->get_items().size());
    ASSERT_GT(count_with_shell, count_before_fire) << "shell must appear after fire";

    // nb_frames_alive = 20 / freq = 20 / (1/60) = 1200 frames
    // run well past that
    for (int i = 0; i < 1300; i++) engine->step(1.f / 60.f);

    const int count_after = static_cast<int>(engine->get_items().size());

    ASSERT_EQ(count_after, count_before_fire) << "shell should be destroyed after lifetime";
}

TEST_F(ShellTest, ShellHitsEnemyTank) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    ASSERT_TRUE(shared_a->has_hit_other_tank()) << "shell should hit the enemy tank";
}

TEST_F(ShellTest, ShellDestroyedOnContact) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const int count_before_fire = static_cast<int>(engine->get_items().size());

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    engine->step(1.f / 60.f);
    ASSERT_GT(static_cast<int>(engine->get_items().size()), count_before_fire)
        << "shell should exist after fire";

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    ASSERT_TRUE(shared_a->has_hit_other_tank()) << "shell must hit enemy tank";

    ASSERT_EQ(static_cast<int>(engine->get_items().size()), count_before_fire)
        << "shell should be destroyed after contact (on_contact self-destructs)";
}

TEST_F(ShellTest, NoFireNoNewItems) {
    add_ground();
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    engine->step(1.f / 60.f);

    const int count_before = static_cast<int>(engine->get_items().size());

    constexpr user_input no_fire{{0.f, 1.f}, {0.f, 0.f}, {false}};
    for (const auto &ctrl: tank->get_controllers()) ctrl->apply_input(no_fire);

    engine->step(1.f / 60.f);

    const int count_after = static_cast<int>(engine->get_items().size());

    ASSERT_EQ(count_after, count_before);
}

TEST_F(ShellTest, ShellContactCallbackSetsReward) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    const std::vector<std::shared_ptr<EnemyTank>> tanks{shared_a, shared_b};

    ASSERT_TRUE(shared_a->has_hit_other_tank()) << "shell must hit for reward test";

    const float reward = shared_a->get_reward(tanks);
    ASSERT_GT(reward, 0.f) << "reward should be positive after shell contact callback";
}

TEST_F(ShellTest, ActionStatsTracksFireButton) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());

    constexpr user_input fire_input{{0.f, 0.f}, {0.f, 0.f}, {true}};
    shared_tank->get_action_stats()->process_input(fire_input);

    ASSERT_TRUE(shared_tank->get_action_stats()->has_fire());
}
