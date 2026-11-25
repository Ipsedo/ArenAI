#ifndef ARENAI_GUN_LAYING_H
#define ARENAI_GUN_LAYING_H

#include <glm/glm.hpp>
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/Body.h>

namespace arenai::model {

    glm::mat3 rotation_of(const JPH::Body &body);

    // stabilised gun: the aim set point is held in the world frame, the hinges follow it
    class GunLaying {
    public:
        GunLaying();

        // positive deltas turn the set point the way a positive hinge angle does
        void rotate_turret(float delta);
        void rotate_canon(float delta);

        // the hinge angles pointing the canon along the set point, from the parent body rotation
        float turret_angle(const glm::mat3 &chassis_rotation) const;
        float canon_angle(const glm::mat3 &turret_rotation) const;

        // the set point falls back on the hinge angle actually reached
        void anchor_turret(const glm::mat3 &chassis_rotation, float turret_angle);
        void anchor_canon(const glm::mat3 &turret_rotation, float canon_angle);

        void reset(const glm::vec3 &canon_forward);

    private:
        float yaw;
        float elevation;

        glm::vec3 direction() const;
        void set_direction(const glm::vec3 &world_direction);
    };

}// namespace arenai::model

#endif// ARENAI_GUN_LAYING_H
