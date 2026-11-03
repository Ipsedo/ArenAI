//
// Created by claude on 03/11/2026.
//

#include <cmath>

#include <gtest/gtest.h>

#include "tank/ballistics.h"

using namespace arenai;
using namespace arenai::model;

TEST(Ballistics, StartsAtTheMuzzleAndSamplesTheWholeDuration) {
    const auto points = predict_shell_trajectory(
        glm::vec3(1.f, 2.f, 3.f), glm::vec3(0.f, 0.f, 1.f), 250.f, 9.8f, 2.f, 0.1f);

    ASSERT_EQ(points.size(), 21);
    ASSERT_FLOAT_EQ(points.front().x, 1.f);
    ASSERT_FLOAT_EQ(points.front().y, 2.f);
    ASSERT_FLOAT_EQ(points.front().z, 3.f);
}

TEST(Ballistics, HorizontalShotFallsUnderGravity) {
    const auto points =
        predict_shell_trajectory(glm::vec3(0.f), glm::vec3(0.f, 0.f, 1.f), 250.f, 9.8f, 1.f, 0.1f);

    const auto &last = points.back();
    ASSERT_NEAR(last.z, 250.f, 1e-3f);
    ASSERT_NEAR(last.y, -0.5f * 9.8f, 1e-3f);
    ASSERT_NEAR(last.x, 0.f, 1e-5f);
}

TEST(Ballistics, DirectionIsNormalised) {
    const auto unit =
        predict_shell_trajectory(glm::vec3(0.f), glm::vec3(0.f, 0.f, 1.f), 100.f, 0.f, 1.f, 0.5f);
    const auto scaled =
        predict_shell_trajectory(glm::vec3(0.f), glm::vec3(0.f, 0.f, 7.f), 100.f, 0.f, 1.f, 0.5f);

    ASSERT_EQ(unit.size(), scaled.size());
    for (size_t i = 0; i < unit.size(); i++) ASSERT_NEAR(unit[i].z, scaled[i].z, 1e-4f);
}

TEST(Ballistics, FortyFiveDegreesApexMatchesTheClosedForm) {
    constexpr float speed = 100.f, gravity = 9.8f;
    const float vertical = speed / std::sqrt(2.f);
    const float apex_time = vertical / gravity;

    const auto points = predict_shell_trajectory(
        glm::vec3(0.f), glm::normalize(glm::vec3(0.f, 1.f, 1.f)), speed, gravity, 2.f * apex_time,
        apex_time / 50.f);

    float max_y = -1.f;
    for (const auto &p: points) max_y = std::max(max_y, p.y);

    ASSERT_NEAR(max_y, vertical * vertical / (2.f * gravity), 1e-2f);
}
