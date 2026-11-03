//
// Created by claude on 03/11/2026.
//

#ifndef ARENAI_BALLISTICS_H
#define ARENAI_BALLISTICS_H

#include <vector>

#include <glm/glm.hpp>

namespace arenai::model {

    std::vector<glm::vec3> predict_shell_trajectory(
        const glm::vec3 &start, const glm::vec3 &direction, float speed, float gravity,
        float duration, float time_step);

}// namespace arenai::model

#endif// ARENAI_BALLISTICS_H
