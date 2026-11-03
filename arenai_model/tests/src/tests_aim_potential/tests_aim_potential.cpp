//
// Created by claude on 03/11/2026.
//

#include <memory>
#include <vector>

#include <arenai_model/item_factory.h>
#include <arenai_model_tests/tests_aim_potential/tests_aim_potential.h>

using namespace arenai;
using namespace arenai::model;

namespace {

    constexpr float FREQUENCY = 1.f / 60.f;

    std::vector<std::shared_ptr<EnemyTank>>
    settle(std::vector<std::unique_ptr<EnemyTank>> tanks, AbstractPhysicEngine &engine) {
        for (int i = 0; i < 300; i++) engine.step(FREQUENCY);

        std::vector<std::shared_ptr<EnemyTank>> shared;
        for (auto &tank: tanks) shared.emplace_back(tank.release());
        return shared;
    }

}// namespace

TEST_F(AimPotentialTest, EnemyInTheAxisScoresHigh) {
    add_ground();
    std::vector<std::unique_ptr<EnemyTank>> tanks;
    tanks.push_back(tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, false));
    tanks.push_back(tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 30.f}, false));
    const auto shared = settle(std::move(tanks), *engine);

    const float potential = shared[0]->get_aim_potential(shared);

    ASSERT_GT(potential, 0.5f);
    ASSERT_LE(potential, 1.f);
}

TEST_F(AimPotentialTest, EnemyOffTheAxisScoresLow) {
    add_ground();
    std::vector<std::unique_ptr<EnemyTank>> tanks;
    tanks.push_back(tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, false));
    tanks.push_back(tank_factory->make_enemy_tank(file_reader, "tank_b", {15.f, 5.f, 30.f}, false));
    const auto shared = settle(std::move(tanks), *engine);

    const float potential = shared[0]->get_aim_potential(shared);

    ASSERT_GE(potential, 0.f);
    ASSERT_LT(potential, 0.5f);
}

TEST_F(AimPotentialTest, ObstacleInTheWayBlocksTheShot) {
    add_ground();
    std::vector<std::unique_ptr<EnemyTank>> tanks;
    tanks.push_back(tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, false));
    tanks.push_back(tank_factory->make_enemy_tank(file_reader, "tank_b", {0.f, 5.f, 60.f}, false));
    const auto shared = settle(std::move(tanks), *engine);

    const float clear = shared[0]->get_aim_potential(shared);

    const auto wall = engine->get_item_factory()->make_cube_item(
        "wall", file_reader, glm::vec3(0.f, 6.f, 30.f), glm::vec3(10.f, 6.f, 1.f), 0.f);
    engine->step(FREQUENCY);

    const float blocked = shared[0]->get_aim_potential(shared);

    ASSERT_GT(clear, 0.5f);
    ASSERT_LT(blocked, 0.5f);
    ASSERT_LT(blocked, clear);
}

TEST_F(AimPotentialTest, NoLivingEnemyScoresZero) {
    add_ground();
    std::vector<std::unique_ptr<EnemyTank>> tanks;
    tanks.push_back(tank_factory->make_enemy_tank(file_reader, "tank_a", {0.f, 5.f, 0.f}, false));
    const auto shared = settle(std::move(tanks), *engine);

    ASSERT_FLOAT_EQ(shared[0]->get_aim_potential(shared), 0.f);
}
