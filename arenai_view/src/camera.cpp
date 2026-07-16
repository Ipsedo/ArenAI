//
// Created by samuel on 18/03/2023.
//

#include <algorithm>
#include <limits>
#include <utility>

#include <arenai_view/camera.h>

namespace arenai::view {

    StaticCamera::StaticCamera(const glm::vec3 pos, const glm::vec3 look, const glm::vec3 up)
        : pos_vec(pos), look_vec(look), up_vec(up) {}

    glm::vec3 StaticCamera::pos() { return pos_vec; }

    glm::vec3 StaticCamera::look() { return look_vec; }

    glm::vec3 StaticCamera::up() { return up_vec; }

    CollisionCamera::CollisionCamera(
        std::shared_ptr<AbstractCamera> inner, RaycastFunction raycast, const float frame_period,
        const float margin, const float min_distance, const float extend_speed)
        : inner(std::move(inner)), raycast(std::move(raycast)), frame_period(frame_period),
          margin(margin), min_distance(min_distance), extend_speed(extend_speed),
          current_distance(std::numeric_limits<float>::max()) {}

    glm::vec3 CollisionCamera::pos() {
        const glm::vec3 pivot = inner->look();
        const glm::vec3 desired = inner->pos();

        const glm::vec3 offset = desired - pivot;
        const float full_distance = glm::length(offset);
        if (full_distance <= 1e-6f) return desired;
        const glm::vec3 dir = offset / full_distance;

        float target = full_distance;
        if (const auto hit_fraction = raycast(pivot, desired); hit_fraction.has_value())
            target = *hit_fraction * full_distance - margin;
        target = std::clamp(target, std::min(min_distance, full_distance), full_distance);

        if (target < current_distance) current_distance = target;
        else
            current_distance +=
                (target - current_distance) * std::min(1.f, extend_speed * frame_period);

        return pivot + dir * current_distance;
    }

    glm::vec3 CollisionCamera::look() { return inner->look(); }

    glm::vec3 CollisionCamera::up() { return inner->up(); }

}// namespace arenai::view
