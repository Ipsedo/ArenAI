//
// Created by claude on 03/11/2026.
//

#ifndef ARENAI_AIM_POTENTIAL_H
#define ARENAI_AIM_POTENTIAL_H

#include <vector>

#include <glm/glm.hpp>
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>

#include "./aim_scorer.h"

namespace arenai::model {

    class JoltPhysicEngine;

    class AimPotential {
    public:
        AimPotential(
            const JoltPhysicEngine &engine, AimScorer scorer, float shell_speed, float gravity,
            float max_flight_time, float time_step);

        float evaluate(
            const glm::vec3 &fire_pos, const glm::vec3 &direction,
            const std::vector<glm::vec3> &targets, const std::vector<JPH::BodyID> &excluded) const;

    private:
        const JoltPhysicEngine &engine;
        AimScorer scorer;
        float shell_speed;
        float gravity;
        float max_flight_time;
        float time_step;
    };

}// namespace arenai::model

#endif// ARENAI_AIM_POTENTIAL_H
