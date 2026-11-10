#include <cmath>
#include <memory>
#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_model/constants.h>
#include <arenai_model/shell_economy.h>
#include <arenai_model_tests/tests_proprioception/tests_proprioception.h>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;

TEST_F(ProprioceptionTest, ProprioceptionSizeCorrect) {
    add_ground();
    auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);

    engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const auto proprio = shared_tank->get_proprioception();

    ASSERT_EQ(proprio.size(), ENEMY_PROPRIOCEPTION_SIZE);
}

TEST_F(ProprioceptionTest, ProprioceptionNoNaN) {
    add_ground();
    auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);

    engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const auto proprio = shared_tank->get_proprioception();

    for (int i = 0; i < proprio.size(); i++) {
        ASSERT_FALSE(std::isnan(proprio[i])) << "NaN at index " << i;
        ASSERT_FALSE(std::isinf(proprio[i])) << "Inf at index " << i;
    }
}

TEST_F(ProprioceptionTest, ProprioceptionNoInfinity) {
    add_ground();
    auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);

    for (int i = 0; i < 10; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const auto proprio = shared_tank->get_proprioception();

    for (int i = 0; i < proprio.size(); i++) {
        ASSERT_FALSE(std::isinf(proprio[i])) << "Inf at index " << i;
    }
}

TEST_F(ProprioceptionTest, ProprioceptionContainsSubItemRelativePositions) {
    add_ground();
    auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);

    engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const auto proprio = shared_tank->get_proprioception();

    bool any_non_zero_position = false;
    const int items_count = static_cast<int>(shared_tank->get_items().size());
    for (int i = 1; i < items_count; i++) {
        const int offset = 12 + (i - 1) * 15;
        const float px = proprio[offset];
        const float py = proprio[offset + 1];
        const float pz = proprio[offset + 2];
        if (std::abs(px) > 1e-3f || std::abs(py) > 1e-3f || std::abs(pz) > 1e-3f) {
            any_non_zero_position = true;
            break;
        }
    }

    ASSERT_TRUE(any_non_zero_position)
        << "sub-items should have non-zero relative positions to chassis";
}

TEST_F(ProprioceptionTest, ProprioceptionConsistentSize) {
    add_ground();
    auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);

    engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());

    const auto proprio_1 = shared_tank->get_proprioception();

    engine->step(1.f / 60.f);

    const auto proprio_2 = shared_tank->get_proprioception();

    ASSERT_EQ(proprio_1.size(), proprio_2.size());
}

TEST_F(ProprioceptionTest, ProprioceptionForwardAndUpVectorsValid) {
    add_ground();
    auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 0.f, 0.f}, GAME_SHELL_ECONOMY);

    engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const auto proprio = shared_tank->get_proprioception();

    // elements [3..5] are chassis forward direction, [6..8] are chassis up direction
    const float fx = proprio[3], fy = proprio[4], fz = proprio[5];
    const float ux = proprio[6], uy = proprio[7], uz = proprio[8];

    const float forward_len = std::sqrt(fx * fx + fy * fy + fz * fz);
    const float up_len = std::sqrt(ux * ux + uy * uy + uz * uz);

    ASSERT_GT(forward_len, 0.1f) << "chassis forward should be non-zero";
    ASSERT_GT(up_len, 0.1f) << "chassis up should be non-zero";

    ASSERT_GT(uy, std::abs(ux)) << "chassis up Y should dominate X";
    ASSERT_GT(uy, std::abs(uz)) << "chassis up Y should dominate Z";
}

TEST_F(ProprioceptionTest, CanFireFollowsCooldownWithoutConsumingIt) {
    add_ground();
    auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());

    ASSERT_TRUE(shared_tank->get_can_fire());
    ASSERT_TRUE(shared_tank->get_can_fire());

    constexpr controller::user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->apply_input(fire_input);

    ASSERT_FALSE(shared_tank->get_can_fire()) << "the shot just fired starts the cooldown";

    const std::vector tanks{shared_tank};
    for (int i = 0; i < 60; i++) {
        engine->step(1.f / 60.f);
        shared_tank->tick(tanks);
    }

    ASSERT_TRUE(shared_tank->get_can_fire()) << "the cooldown is over and shells remain";
}

TEST_F(ProprioceptionTest, ShellAndCooldownRatiosFollowAShot) {
    add_ground();
    auto tank =
        tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, GAME_SHELL_ECONOMY);

    for (int i = 0; i < 300; i++) engine->step(1.f / 60.f);

    const std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const std::vector tanks{shared_tank};

    constexpr float initial_reserve = static_cast<float>(GAME_SHELL_ECONOMY.initial_shells)
                                      / static_cast<float>(GAME_SHELL_ECONOMY.max_shells);
    ASSERT_FLOAT_EQ(shared_tank->get_remaining_shells_ratio(), initial_reserve);
    ASSERT_FLOAT_EQ(shared_tank->get_cooldown_ratio(), 1.f) << "a fresh canon is ready";

    constexpr controller::user_input fire_input{
        .left_joystick = {.x = 0.f, .y = 0.f},
        .right_joystick = {.x = 0.f, .y = 0.f},
        .fire_button = {true}};
    for (const auto &ctrl: shared_tank->get_controllers()) ctrl->apply_input(fire_input);
    ASSERT_FLOAT_EQ(shared_tank->get_cooldown_ratio(), 0.f) << "the shot starts the cooldown";

    engine->step(1.f / 60.f);
    shared_tank->tick(tanks);

    ASSERT_FLOAT_EQ(
        shared_tank->get_remaining_shells_ratio(),
        initial_reserve - 1.f / static_cast<float>(GAME_SHELL_ECONOMY.max_shells));
    ASSERT_LT(shared_tank->get_cooldown_ratio(), 1.f) << "the canon is still cooling down";

    for (int i = 0; i < 60; i++) {
        engine->step(1.f / 60.f);
        shared_tank->tick(tanks);
    }

    ASSERT_FLOAT_EQ(shared_tank->get_cooldown_ratio(), 1.f) << "the cooldown is over";
}
