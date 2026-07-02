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
