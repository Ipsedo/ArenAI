//
// Created by claude on 03/11/2026.
//

#ifndef ARENAI_GEOMETRY_H
#define ARENAI_GEOMETRY_H

#include <algorithm>

#include <glm/glm.hpp>

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/norm.hpp>

namespace arenai::model {

    inline float distance_to_segment(
        const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &p, glm::vec3 &closest) {
        const glm::vec3 ab = b - a;
        const float length_squared = glm::length2(ab);

        const float t =
            length_squared > 0.f ? std::clamp(glm::dot(p - a, ab) / length_squared, 0.f, 1.f) : 0.f;

        closest = a + t * ab;

        return glm::length(p - closest);
    }

}// namespace arenai::model

#endif// ARENAI_GEOMETRY_H
