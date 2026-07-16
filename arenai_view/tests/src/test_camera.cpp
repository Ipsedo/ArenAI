//
// Created by claude on 01/07/2026.
//

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <arenai_view/camera.h>

using namespace arenai;
using namespace arenai::view;

TEST(StaticCameraTest, ReturnsConstructorValues) {
    constexpr glm::vec3 pos{1.f, 2.f, 3.f};
    constexpr glm::vec3 look{4.f, 5.f, 6.f};
    constexpr glm::vec3 up{0.f, 1.f, 0.f};

    StaticCamera camera(pos, look, up);

    ASSERT_EQ(camera.pos(), pos);
    ASSERT_EQ(camera.look(), look);
    ASSERT_EQ(camera.up(), up);
}

TEST(StaticCameraTest, NegativeCoordinates) {
    constexpr glm::vec3 pos{-10.f, -20.f, -30.f};
    constexpr glm::vec3 look{-1.f, -2.f, -3.f};
    constexpr glm::vec3 up{0.f, -1.f, 0.f};

    StaticCamera camera(pos, look, up);

    ASSERT_EQ(camera.pos(), pos);
    ASSERT_EQ(camera.look(), look);
    ASSERT_EQ(camera.up(), up);
}

TEST(StaticCameraTest, ZeroVectors) {
    constexpr glm::vec3 zero{0.f, 0.f, 0.f};

    StaticCamera camera(zero, zero, zero);

    ASSERT_EQ(camera.pos(), zero);
    ASSERT_EQ(camera.look(), zero);
    ASSERT_EQ(camera.up(), zero);
}

TEST(StaticCameraTest, LargeValues) {
    constexpr glm::vec3 pos{1e6f, 1e6f, 1e6f};
    constexpr glm::vec3 look{-1e6f, -1e6f, -1e6f};
    constexpr glm::vec3 up{0.f, 1.f, 0.f};

    StaticCamera camera(pos, look, up);

    ASSERT_EQ(camera.pos(), pos);
    ASSERT_EQ(camera.look(), look);
    ASSERT_EQ(camera.up(), up);
}

// pivot (= look) at origin, desired camera position 20 units behind on -Z
constexpr glm::vec3 COLLISION_PIVOT{0.f, 0.f, 0.f};
constexpr glm::vec3 COLLISION_DESIRED{0.f, 0.f, -20.f};
constexpr glm::vec3 COLLISION_UP{0.f, 1.f, 0.f};
constexpr float FULL_DISTANCE = 20.f;
constexpr float FRAME_PERIOD = 1.f / 60.f;
constexpr float MARGIN = 0.5f;
constexpr float MIN_DISTANCE = 2.f;
constexpr float EXTEND_SPEED = 4.f;

static std::shared_ptr<AbstractCamera> make_inner() {
    return std::make_shared<StaticCamera>(COLLISION_DESIRED, COLLISION_PIVOT, COLLISION_UP);
}

static CollisionCamera make_collision_camera(RaycastFunction raycast) {
    return {make_inner(), std::move(raycast), FRAME_PERIOD, MARGIN, MIN_DISTANCE, EXTEND_SPEED};
}

TEST(CollisionCameraTest, ForwardsLookAndUp) {
    auto camera = make_collision_camera([](glm::vec3, glm::vec3) { return std::nullopt; });

    ASSERT_EQ(camera.look(), COLLISION_PIVOT);
    ASSERT_EQ(camera.up(), COLLISION_UP);
}

TEST(CollisionCameraTest, FreePathKeepsDesiredPosition) {
    auto camera = make_collision_camera([](glm::vec3, glm::vec3) { return std::nullopt; });

    const glm::vec3 pos = camera.pos();
    ASSERT_NEAR(pos.z, COLLISION_DESIRED.z, 1e-5f);
    ASSERT_NEAR(pos.x, COLLISION_DESIRED.x, 1e-5f);
    ASSERT_NEAR(pos.y, COLLISION_DESIRED.y, 1e-5f);
}

TEST(CollisionCameraTest, RaycastReceivesPivotAndDesired) {
    glm::vec3 received_from{}, received_to{};
    auto camera = make_collision_camera(
        [&](const glm::vec3 from, const glm::vec3 to) -> std::optional<float> {
            received_from = from;
            received_to = to;
            return std::nullopt;
        });

    camera.pos();

    ASSERT_EQ(received_from, COLLISION_PIVOT);
    ASSERT_EQ(received_to, COLLISION_DESIRED);
}

TEST(CollisionCameraTest, ObstacleAtMidwayPullsCameraInFrontOfIt) {
    auto camera = make_collision_camera([](glm::vec3, glm::vec3) { return 0.5f; });

    const glm::vec3 pos = camera.pos();
    // 0.5 * 20 - margin along -Z from the pivot
    ASSERT_NEAR(glm::length(pos - COLLISION_PIVOT), 0.5f * FULL_DISTANCE - MARGIN, 1e-5f);
    ASSERT_LT(pos.z, 0.f);
}

TEST(CollisionCameraTest, ObstacleNextToPivotClampsToMinDistance) {
    auto camera = make_collision_camera([](glm::vec3, glm::vec3) { return 0.01f; });

    const glm::vec3 pos = camera.pos();
    ASSERT_NEAR(glm::length(pos - COLLISION_PIVOT), MIN_DISTANCE, 1e-5f);
}

TEST(CollisionCameraTest, RetractionIsInstantaneous) {
    float hit_fraction = 1.f;
    auto camera = make_collision_camera(
        [&](glm::vec3, glm::vec3) -> std::optional<float> { return hit_fraction; });

    camera.pos();// settled at full distance - margin

    hit_fraction = 0.25f;
    const glm::vec3 pos = camera.pos();
    ASSERT_NEAR(glm::length(pos - COLLISION_PIVOT), 0.25f * FULL_DISTANCE - MARGIN, 1e-5f);
}

TEST(CollisionCameraTest, ExtensionIsSmoothedAndMonotonic) {
    bool blocked = true;
    auto camera = make_collision_camera([&](glm::vec3, glm::vec3) -> std::optional<float> {
        if (blocked) return 0.5f;
        return std::nullopt;
    });

    const float blocked_distance = glm::length(camera.pos() - COLLISION_PIVOT);

    blocked = false;
    float previous = blocked_distance;
    for (int i = 0; i < 10; i++) {
        const float distance = glm::length(camera.pos() - COLLISION_PIVOT);
        ASSERT_GT(distance, previous);
        ASSERT_LE(distance, FULL_DISTANCE + 1e-5f);
        previous = distance;
    }

    // one smoothed step must not jump straight back to the nominal distance
    ASSERT_LT(previous, FULL_DISTANCE);
    const float first_step = EXTEND_SPEED * FRAME_PERIOD;
    ASSERT_LT(
        glm::length(camera.pos() - COLLISION_PIVOT),
        blocked_distance + (FULL_DISTANCE - blocked_distance) * first_step * 12.f);
}
