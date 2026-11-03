//
// Created by claude on 03/11/2026.
//

#include "./ballistics.h"

#include <cmath>

namespace arenai::model {

    std::vector<glm::vec3> predict_shell_trajectory(
        const glm::vec3 &start, const glm::vec3 &direction, const float speed, const float gravity,
        const float duration, const float time_step) {
        const auto nb_steps = static_cast<int>(std::ceil(duration / time_step));

        const glm::vec3 velocity = glm::normalize(direction) * speed;
        const glm::vec3 acceleration(0.f, -gravity, 0.f);

        std::vector<glm::vec3> points;
        points.reserve(nb_steps + 1);

        for (int i = 0; i <= nb_steps; i++) {
            const float t = static_cast<float>(i) * time_step;
            points.push_back(start + velocity * t + 0.5f * acceleration * t * t);
        }

        return points;
    }

}// namespace arenai::model
