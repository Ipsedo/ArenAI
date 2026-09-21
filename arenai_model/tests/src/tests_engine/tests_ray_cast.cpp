//
// Created by claude on 21/09/2026.
//

#include <glm/glm.hpp>

#include <arenai_model_tests/utils/engine_test_fixture.h>

using namespace arenai;
using namespace arenai::model;

// ========================================================================
// ray_cast — hit position, miss cases
// ========================================================================

TEST_F(EngineTestFixture, RayCastHitsGroundWithoutStep) {
    add_ground();

    // no engine->step: spawn code casts rays right after adding bodies
    const auto hit =
        engine->ray_cast(glm::vec3(10.f, 100.f, 10.f), glm::vec3(10.f, -100.f, 10.f));

    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(hit->x, 10.f, 1e-3f);
    EXPECT_NEAR(hit->y, 0.f, 1e-2f);
    EXPECT_NEAR(hit->z, 10.f, 1e-3f);
}

TEST_F(EngineTestFixture, RayCastMissReturnsNullopt) {
    // empty world
    EXPECT_FALSE(
        engine->ray_cast(glm::vec3(0.f, 100.f, 0.f), glm::vec3(0.f, -100.f, 0.f)).has_value());

    add_ground();

    // segment ending above the ground
    EXPECT_FALSE(
        engine->ray_cast(glm::vec3(0.f, 100.f, 0.f), glm::vec3(0.f, 50.f, 0.f)).has_value());
}
