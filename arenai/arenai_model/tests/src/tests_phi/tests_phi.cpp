//
// Created by samuel on 01/07/2026.
//

#include <cmath>
#include <memory>

#include <arenai_model_tests/tests_phi/tests_phi.h>

// ========================================================================
// get_phi (engagement/opportunity score)
// ========================================================================

TEST_F(PhiTest, PhiZeroWithNoOtherTanks) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks;
    tanks.emplace_back(tank.release());

    const float phi = tanks[0]->get_phi(tanks);

    ASSERT_FLOAT_EQ(phi, 0.f);
}

TEST_F(PhiTest, PhiZeroWhenAllOthersDead) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {10.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks;
    tanks.emplace_back(tank_a.release());
    tanks.emplace_back(tank_b.release());

    // kill tank_b
    for (const auto &item: tanks[1]->get_items()) {
        if (auto *life = dynamic_cast<LifeItem *>(item.get())) {
            life->receive_damages(1e6f);
            break;
        }
    }
    ASSERT_TRUE(tanks[1]->is_dead());

    const float phi = tanks[0]->get_phi(tanks);

    ASSERT_FLOAT_EQ(phi, 0.f);
}

TEST_F(PhiTest, PhiPositiveWithLiveEnemy) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {10.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks;
    tanks.emplace_back(tank_a.release());
    tanks.emplace_back(tank_b.release());

    const float phi = tanks[0]->get_phi(tanks);

    ASSERT_GT(phi, 0.f);
}

TEST_F(PhiTest, PhiBoundedZeroToOne) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {5.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks;
    tanks.emplace_back(tank_a.release());
    tanks.emplace_back(tank_b.release());

    const float phi = tanks[0]->get_phi(tanks);

    ASSERT_GE(phi, 0.f);
    ASSERT_LE(phi, 1.f);
}

TEST_F(PhiTest, PhiHigherWhenEnemyCloser) {
    add_ground();
    auto tank_close = tank_factory->make_enemy_tank("tank_close", {0.f, 0.f, 0.f});
    auto enemy_close = tank_factory->make_enemy_tank("enemy_close", {0.f, 0.f, 15.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks_close;
    tanks_close.emplace_back(tank_close.release());
    tanks_close.emplace_back(enemy_close.release());

    const float phi_close = tanks_close[0]->get_phi(tanks_close);

    // clean up and recreate with far enemy along same axis
    engine->remove_bodies_and_constraints();
    engine = make_physic_engine(60.f);
    tank_factory = make_tank_factory(*engine, file_reader, 60.f);

    add_ground();
    auto tank_far = tank_factory->make_enemy_tank("tank_far", {0.f, 0.f, 0.f});
    auto enemy_far = tank_factory->make_enemy_tank("enemy_far", {0.f, 0.f, 500.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks_far;
    tanks_far.emplace_back(tank_far.release());
    tanks_far.emplace_back(enemy_far.release());

    const float phi_far = tanks_far[0]->get_phi(tanks_far);

    ASSERT_GT(phi_close, phi_far);
}

TEST_F(PhiTest, PhiIgnoresSelf) {
    add_ground();
    auto tank = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});

    engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks;
    tanks.emplace_back(tank.release());

    const float phi = tanks[0]->get_phi(tanks);

    ASSERT_FLOAT_EQ(phi, 0.f);
}

TEST_F(PhiTest, PhiNoNaN) {
    add_ground();
    auto tank_a = tank_factory->make_enemy_tank("tank_a", {0.f, 0.f, 0.f});
    auto tank_b = tank_factory->make_enemy_tank("tank_b", {15.f, 0.f, 0.f});

    for (int i = 0; i < 10; i++) engine->step(1.f / 60.f);

    std::vector<std::shared_ptr<EnemyTank>> tanks;
    tanks.emplace_back(tank_a.release());
    tanks.emplace_back(tank_b.release());

    const float phi = tanks[0]->get_phi(tanks);

    ASSERT_FALSE(std::isnan(phi));
    ASSERT_FALSE(std::isinf(phi));
}
