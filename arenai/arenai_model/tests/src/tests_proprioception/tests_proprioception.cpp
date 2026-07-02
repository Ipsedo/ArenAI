//
// Created by samuel on 01/07/2026.
//

#include <cmath>
#include <memory>

#include <arenai_model/constants.h>
#include <arenai_model_tests/tests_proprioception/tests_proprioception.h>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;

// ========================================================================
// Proprioception
// ========================================================================

TEST_F(ProprioceptionTest, ProprioceptionSizeCorrect) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const auto proprio = shared_tank->get_proprioception();

    ASSERT_EQ(proprio.size(), ENEMY_PROPRIOCEPTION_SIZE);
}

TEST_F(ProprioceptionTest, ProprioceptionNoNaN) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const auto proprio = shared_tank->get_proprioception();

    for (int i = 0; i < proprio.size(); i++) {
        ASSERT_FALSE(std::isnan(proprio[i])) << "NaN at index " << i;
        ASSERT_FALSE(std::isinf(proprio[i])) << "Inf at index " << i;
    }
}

TEST_F(ProprioceptionTest, ProprioceptionNoInfinity) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});

    for (int i = 0; i < 10; i++) engine->step(1.f / 60.f);

    std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const auto proprio = shared_tank->get_proprioception();

    for (int i = 0; i < proprio.size(); i++) {
        ASSERT_FALSE(std::isinf(proprio[i])) << "Inf at index " << i;
    }
}

TEST_F(ProprioceptionTest, ProprioceptionContainsSubItemRelativePositions) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const auto proprio = shared_tank->get_proprioception();

    // after the chassis data (12 floats), each sub-item contributes 15 floats:
    // pos(3) + vel(3) + forward(3) + up(3) + ang_vel(3)
    // the relative positions of wheels/turret/canon should not all be zero
    // (they are at different positions relative to chassis)
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
    auto tank = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::shared_ptr<EnemyTank> shared_tank(tank.release());

    const auto proprio_1 = shared_tank->get_proprioception();

    engine->step(1.f / 60.f);

    const auto proprio_2 = shared_tank->get_proprioception();

    ASSERT_EQ(proprio_1.size(), proprio_2.size());
}

TEST_F(ProprioceptionTest, ProprioceptionForwardAndUpVectorsValid) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::shared_ptr<EnemyTank> shared_tank(tank.release());
    const auto proprio = shared_tank->get_proprioception();

    // elements [3..5] are chassis forward direction, [6..8] are chassis up direction
    const float fx = proprio[3], fy = proprio[4], fz = proprio[5];
    const float ux = proprio[6], uy = proprio[7], uz = proprio[8];

    const float forward_len = std::sqrt(fx * fx + fy * fy + fz * fz);
    const float up_len = std::sqrt(ux * ux + uy * uy + uz * uz);

    // forward and up lengths are affected by the model matrix scale (0.5),
    // so they are not necessarily unit vectors — just verify they are non-zero
    ASSERT_GT(forward_len, 0.1f) << "chassis forward should be non-zero";
    ASSERT_GT(up_len, 0.1f) << "chassis up should be non-zero";

    // for a tank at rest, up direction should point mostly upward (Y dominant)
    // the Y component is scaled by 0.5 (tank scale), so check relative magnitude
    ASSERT_GT(uy, std::abs(ux)) << "chassis up Y should dominate X";
    ASSERT_GT(uy, std::abs(uz)) << "chassis up Y should dominate Z";
}
