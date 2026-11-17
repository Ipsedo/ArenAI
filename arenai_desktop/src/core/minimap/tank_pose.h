#ifndef ARENAI_DESKTOP_CORE_MINIMAP_TANK_POSE_H
#define ARENAI_DESKTOP_CORE_MINIMAP_TANK_POSE_H

#include <glm/glm.hpp>

namespace arenai::desktop {

    struct TankPose {
        glm::vec3 position;
        // the chassis' forward axis, world space
        glm::vec3 forward;
        bool alive = true;
    };

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_MINIMAP_TANK_POSE_H
