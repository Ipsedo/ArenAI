#include <cmath>
#include <memory>
#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_model/constants.h>
#include <arenai_model/item_factory.h>
#include <arenai_model/shell_economy.h>
#include <arenai_model_tests/tests_reward/tests_reward.h>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;
using namespace arenai::controller;

TEST_F(RewardTest, RewardZeroWhenAliveNoShot) {
    add_ground();
    auto tank_a =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);
    auto tank_b =
        tank_factory->make_enemy_tank(file_reader, "tank_b", {20.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);

    engine->step(1.f / 60.f);

    const std::vector tanks{
        std::shared_ptr<EnemyTank>(tank_a.release()), std::shared_ptr<EnemyTank>(tank_b.release())};

    const float reward_a = tanks[0]->get_reward();
    const float reward_b = tanks[1]->get_reward();

    ASSERT_FLOAT_EQ(reward_a, 0.f);
    ASSERT_FLOAT_EQ(reward_b, 0.f);
}

TEST_F(RewardTest, RewardNegativeWhenDead) {
    add_ground();
    auto tank_a =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);
    auto tank_b =
        tank_factory->make_enemy_tank(file_reader, "tank_b", {20.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);

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
    auto tank_a =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);
    auto tank_b =
        tank_factory->make_enemy_tank(file_reader, "tank_b", {20.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);

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
    const auto detail = tanks[1]->get_last_reward_detail();

    ASSERT_FLOAT_EQ(detail.death, -DEATH_PENALTY);
    ASSERT_FLOAT_EQ(detail.received, -DAMAGE_REWARD);
    ASSERT_FLOAT_EQ(death_reward, -DEATH_PENALTY - DAMAGE_REWARD);
}

TEST_F(RewardTest, RewardPositiveOnHit) {
    add_ground();
    // spawn tanks high enough so all parts start above ground and settle cleanly
    auto tank_a =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto tank_b =
        tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f}, GAME_SHELL_ECONOMY);

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

    ASSERT_GE(max_reward, 0.1f)
        << "a landed shell pays the health fraction it removed, 0.1 on a chassis, 0.2 elsewhere";
}

TEST_F(RewardTest, RewardUnderOneAfterHit) {
    add_ground();
    // spawn tanks high enough so all parts start above ground and settle cleanly
    auto tank_a =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto tank_b =
        tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f}, GAME_SHELL_ECONOMY);

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

    ASSERT_GE(max_reward_on_hit, 0.1f)
        << "a landed shell pays the health fraction it removed, 0.1 on a chassis, 0.2 elsewhere";

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

    ASSERT_FLOAT_EQ(max_reward_on_no_hit, 0.f)
        << "reward should be exactly zero when no shell hit an enemy";
}

TEST_F(RewardTest, MissedShellPaysNothing) {
    add_ground();
    auto tank_a =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto tank_b =
        tank_factory->make_enemy_tank(file_reader, "tank_b", {15.f, 5.f, 30.f}, GAME_SHELL_ECONOMY);

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    const std::vector tanks{shared_a, shared_b};

    float total_reward = 0.f;
    RewardDetail fire_detail, landed_detail;
    int fire_frame = -1;
    for (int i = 0; i < 180; i++) {
        engine->step(1.f / 60.f);
        shared_a->tick(tanks);

        total_reward += shared_a->get_reward();
        const auto detail = shared_a->get_last_reward_detail();
        if (detail.nb_fired_shells > 0) {
            fire_detail = detail;
            fire_frame = i;
        }
        if (detail.nb_landed_shells > 0) landed_detail = detail;
    }

    ASSERT_FALSE(shared_a->consume_has_hit()) << "shell should have missed the enemy tank";
    ASSERT_EQ(fire_frame, 0) << "the fired shell is reported on the fire frame";
    ASSERT_EQ(fire_detail.nb_fired_shells, 1);
    ASSERT_EQ(landed_detail.nb_landed_shells, 1)
        << "the shell tracker should have sampled a landed shell";
    ASSERT_EQ(landed_detail.nb_hits, 0);
    ASSERT_FLOAT_EQ(landed_detail.hit, 0.f);
    ASSERT_FLOAT_EQ(landed_detail.kill, 0.f);
    ASSERT_FLOAT_EQ(total_reward, 0.f) << "the model's reward is the game's objective only";
}

TEST_F(RewardTest, HitPaysTheHealthFractionRemoved) {
    add_ground();
    auto tank_a =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto tank_b =
        tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f}, GAME_SHELL_ECONOMY);

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    constexpr user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_a->get_controllers()) ctrl->apply_input(fire_input);

    const std::vector tanks{shared_a, shared_b};

    float total_reward = 0.f;
    RewardDetail fire_detail, landed_detail;
    for (int i = 0; i < 60; i++) {
        engine->step(1.f / 60.f);
        shared_a->tick(tanks);

        total_reward += shared_a->get_reward();
        const auto detail = shared_a->get_last_reward_detail();
        if (detail.nb_fired_shells > 0) fire_detail = detail;
        if (detail.nb_landed_shells > 0) landed_detail = detail;
    }

    ASSERT_TRUE(shared_a->consume_has_hit()) << "shell should have hit the enemy tank";
    ASSERT_EQ(fire_detail.nb_fired_shells, 1);
    ASSERT_FLOAT_EQ(fire_detail.hit, 0.f) << "the hit is paid at the impact, not at the fire";

    // the touched part is the one that lost health: the hit pays the fraction it lost
    float lost_fraction = 0.f;
    for (const auto &item: shared_b->get_items())
        if (const auto *life = dynamic_cast<LifeItem *>(item.get()))
            lost_fraction += 1.f - life->health_ratio();
    ASSERT_GT(lost_fraction, 0.f);

    ASSERT_EQ(landed_detail.nb_landed_shells, 1);
    ASSERT_EQ(landed_detail.nb_hits, 1);
    ASSERT_FLOAT_EQ(landed_detail.hit, DAMAGE_REWARD * lost_fraction);
    ASSERT_FLOAT_EQ(landed_detail.kill, 0.f) << "a single touch is not a kill";
    ASSERT_FLOAT_EQ(total_reward, landed_detail.hit);

    // the victim is charged the same fraction
    shared_b->get_reward();
    ASSERT_FLOAT_EQ(shared_b->get_last_reward_detail().received, -DAMAGE_REWARD * lost_fraction);
}

TEST_F(RewardTest, NoRewardWhenShootingAWreck) {

    add_ground();
    // spawn tanks high enough so all parts start above ground and settle cleanly
    auto tank_a =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto tank_b =
        tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f}, GAME_SHELL_ECONOMY);

    // settle on ground (300 frames = 5s at 60fps)
    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    for (const auto items_b = shared_b->get_items(); const auto &item: items_b) {
        if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
            life->receive_damages({.impact_position = glm::vec3(0.f), .damages = 1e6f});
            break;
        }
    }
    ASSERT_TRUE(shared_b->is_dead());
    shared_b->on_death();

    const auto shells_ratio_before = shared_a->get_remaining_shells_ratio();

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
    ASSERT_LT(shared_a->get_remaining_shells_ratio(), shells_ratio_before)
        << "a wreck must not recharge the shell reserve";
}

TEST_F(RewardTest, NoKillRewardWhenHittingAnotherPartOfAWreck) {
    add_ground();
    // spawn tanks high enough so all parts start above ground and settle cleanly
    auto tank_a =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);
    auto tank_b =
        tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f}, GAME_SHELL_ECONOMY);

    // settle on ground (300 frames = 5s at 60fps)
    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_a(tank_a.release());
    const std::shared_ptr<EnemyTank> shared_b(tank_b.release());

    for (const auto items_b = shared_b->get_items(); const auto &item: items_b) {
        auto *life = dynamic_cast<LifeItem *>(item.get());
        if (life && item->get_name().find("wheel_right_1") != std::string::npos) {
            life->receive_damages({.impact_position = glm::vec3(0.f), .damages = 1e6f});
            break;
        }
    }
    ASSERT_TRUE(shared_b->is_dead());
    shared_b->on_death();

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

TEST_F(RewardTest, ZeroRewardWithEmptyTankList) {
    add_ground();
    auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);

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

    // there is no enemy to sample: nothing is paid
    ASSERT_FLOAT_EQ(reward, 0.f);
}

class AimPotentialTest : public RewardTest {
protected:
    std::shared_ptr<EnemyTank> spawn(const std::string &name, const glm::vec3 pos) {
        return tank_factory->make_enemy_tank(file_reader, name, pos, GAME_SHELL_ECONOMY);
    }

    void settle() {
        for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);
    }

    static void kill(const std::shared_ptr<EnemyTank> &tank) {
        for (const auto &item: tank->get_items())
            if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
                life->receive_damages(
                    {.fire_position = glm::vec3(0.f),
                     .impact_position = glm::vec3(0.f),
                     .damages = 1e6f});
                break;
            }
        tank->on_death();
    }
};

TEST_F(AimPotentialTest, PeaksOnACloseEnemyInTheAxis) {
    add_ground();
    const auto shooter = spawn("tank_a", {0.f, 5.f, 0.f});
    const auto enemy = spawn("tank_b", {0.f, 5.f, 30.f});
    settle();

    ASSERT_FLOAT_EQ(shooter->get_aim_potential({shooter}), 0.f);

    const float potential = shooter->get_aim_potential({shooter, enemy});
    ASSERT_GT(potential, 1.4f);
    ASSERT_LE(potential, 1.f + AIM_BEARING_WEIGHT);
}

TEST_F(AimPotentialTest, SlopesTowardAnEnemyOnTheFlank) {
    add_ground();
    const auto shooter = spawn("tank_a", {0.f, 5.f, 0.f});
    const auto flank = spawn("tank_b", {30.f, 5.f, 0.f});
    const auto quarter = spawn("tank_c", {21.2f, 5.f, 21.2f});
    const auto behind = spawn("tank_d", {0.f, 5.f, -30.f});
    settle();

    const float on_flank = shooter->get_aim_potential({shooter, flank});
    ASSERT_NEAR(on_flank, 0.5f * AIM_BEARING_WEIGHT, 0.1f);
    ASSERT_GT(shooter->get_aim_potential({shooter, quarter}), on_flank + 0.15f);
    ASSERT_LT(shooter->get_aim_potential({shooter, behind}), 0.1f);
}

TEST_F(AimPotentialTest, FollowsTheMostUrgentEnemy) {
    add_ground();
    const auto shooter = spawn("tank_a", {0.f, 5.f, 0.f});
    const auto near_flank = spawn("tank_b", {40.f, 5.f, 0.f});
    const auto far_axis = spawn("tank_c", {0.f, 5.f, 300.f});
    settle();

    const float near_potential = shooter->get_aim_potential({shooter, near_flank});
    const float far_potential = shooter->get_aim_potential({shooter, far_axis});

    ASSERT_GT(near_potential, far_potential);
    ASSERT_FLOAT_EQ(shooter->get_aim_potential({shooter, near_flank, far_axis}), near_potential);
}

TEST_F(AimPotentialTest, IgnoresWrecks) {
    add_ground();
    const auto shooter = spawn("tank_a", {0.f, 5.f, 0.f});
    const auto enemy = spawn("tank_b", {0.f, 5.f, 30.f});
    settle();

    kill(enemy);
    ASSERT_FLOAT_EQ(shooter->get_aim_potential({shooter, enemy}), 0.f);

    const auto other = spawn("tank_c", {30.f, 5.f, 0.f});
    kill(shooter);
    ASSERT_FLOAT_EQ(shooter->get_aim_potential({shooter, other}), 0.f);
}
