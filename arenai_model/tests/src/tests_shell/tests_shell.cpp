//
// Created by samuel on 01/07/2026.
//

#include <cmath>
#include <memory>

#include <arenai_controller/inputs.h>
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

    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
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

    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
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

    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    ASSERT_TRUE(shared_a->consume_has_hit()) << "shell should hit the enemy tank";
}

TEST_F(ShellTest, ShellDestroyedOnContact) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const int count_before_fire = static_cast<int>(engine->get_items().size());

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    engine->step(1.f / 60.f);
    ASSERT_GT(static_cast<int>(engine->get_items().size()), count_before_fire)
        << "shell should exist after fire";

    for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

    ASSERT_TRUE(shared_a->consume_has_hit()) << "shell must hit enemy tank";

    ASSERT_EQ(static_cast<int>(engine->get_items().size()), count_before_fire)
        << "shell should be destroyed after contact (on_contact self-destructs)";
}

TEST_F(ShellTest, NoFireNoNewItems) {
    add_ground();
    const auto tank = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});

    engine->step(1.f / 60.f);

    const int count_before = static_cast<int>(engine->get_items().size());

    constexpr user_input no_fire{
        .left_joystick = {.x = 0.f, .y = 1.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {false}};
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
        max_reward = std::max(max_reward, shared_a->get_reward(tanks));
    }
    ASSERT_TRUE(shared_a->consume_has_hit()) << "shell must hit for reward test";

    ASSERT_GT(max_reward, 0.f) << "reward should be positive after shell contact callback";
}

// ========================================================================
// ShellItem — damages dealt per impact
// ========================================================================

namespace {

    constexpr user_input FIRE_INPUT{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};

    void fire_once(const std::shared_ptr<EnemyTank> &tank) {
        for (const auto &ctrl: tank->get_controllers()) ctrl->apply_input(FIRE_INPUT);
    }

    // a tank spreads its health over its parts: its damages are the sum over them
    int consume_tank_hits(const std::shared_ptr<EnemyTank> &tank) {
        int hits = 0;
        for (const auto &item: tank->get_items())
            if (const auto life_item = dynamic_cast<LifeItem *>(item.get()); life_item)
                hits += life_item->consume_hits_received();
        return hits;
    }

    // ShellItem lives behind arenai_model's private Jolt headers: match it by name
    std::shared_ptr<Item> find_shell(AbstractPhysicEngine &engine) {
        for (const auto &item: engine.get_items())
            if (item->get_name() == "shell_item") return item;
        return nullptr;
    }

}// namespace

TEST_F(ShellTest, ShellImpactDealsExactlyOneDamage) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    consume_tank_hits(shared_b);// drop whatever the settling produced

    fire_once(shared_a);
    for (int i = 0; i < 60; i++) {
        engine->step(1.f / 60.f);
        shared_a->tick({shared_a, shared_b});
    }
    ASSERT_TRUE(shared_a->consume_has_hit()) << "shell must hit for this test to mean anything";

    ASSERT_EQ(consume_tank_hits(shared_b), 1)
        << "one shell removes exactly one health point, whatever the number of contact "
           "points the physics engine reports for the impact";
}

TEST_F(ShellTest, TankSurvivesUntilAPartsHealthPointsAreSpent) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    consume_tank_hits(shared_b);// drop whatever the settling produced

    // the cheapest part (wheel, turret, canon) is worth 5 health points, and every
    // shell removes at most one: no tank dies in fewer than 5 shells, whatever the
    // parts they land on
    constexpr int cheapest_part_health_points = 5;
    constexpr int max_shots = 20;

    int hits = 0;
    int shots = 0;
    while (!shared_b->is_dead() && shots < max_shots) {
        fire_once(shared_a);
        for (int i = 0; i < 60; i++) engine->step(1.f / 60.f);

        hits += consume_tank_hits(shared_b);
        shots++;
    }

    ASSERT_TRUE(shared_b->is_dead()) << "tank should have died within " << max_shots << " shots";
    ASSERT_GE(shots, cheapest_part_health_points)
        << "tank died in " << shots << " shells for " << hits
        << " health points: a single impact removed more than one";
}

TEST_F(ShellTest, SpentShellDealsNoFurtherDamage) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f});

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    fire_once(shared_a);
    engine->step(1.f / 60.f);

    const auto shell = find_shell(*engine);
    ASSERT_NE(shell, nullptr) << "shell must exist after fire";

    const auto target = shared_b->get_chassis();
    const auto target_life = dynamic_cast<LifeItem *>(target.get());
    ASSERT_NE(target_life, nullptr) << "the chassis is a life item";

    target_life->consume_hits_received();

    // a shell colliding with several bodies in the same step is dispatched once per
    // body: only the first contact may be paid for
    shell->on_contact(target.get());
    shell->on_contact(target.get());

    ASSERT_EQ(target_life->consume_hits_received(), 1)
        << "a spent shell must not damage anything again";
}
