//
// Created by claude on 03/11/2026.
//

#include "./aim_potential.h"

#include <algorithm>
#include <limits>

#include "../geometry.h"
#include "../jolt_engine.h"
#include "./ballistics.h"

namespace arenai::model {

    AimPotential::AimPotential(
        const JoltPhysicEngine &engine, const AimScorer scorer, const float shell_speed,
        const float gravity, const float max_flight_time, const float time_step)
        : engine(engine), scorer(scorer), shell_speed(shell_speed), gravity(gravity),
          max_flight_time(max_flight_time), time_step(time_step) {}

    float AimPotential::evaluate(
        const glm::vec3 &fire_pos, const glm::vec3 &direction,
        const std::vector<glm::vec3> &targets, const std::vector<JPH::BodyID> &excluded) const {
        if (targets.empty()) return 0.f;

        const auto points = predict_shell_trajectory(
            fire_pos, direction, shell_speed, gravity, max_flight_time, time_step);

        std::vector min_distances(targets.size(), std::numeric_limits<float>::infinity());
        std::vector closest_points(targets.size(), fire_pos);

        for (size_t i = 1; i < points.size(); i++) {
            const glm::vec3 &a = points[i - 1];
            glm::vec3 b = points[i];

            const auto hit = engine.ray_test(a, b, excluded);
            if (hit.has_value()) b = a + *hit * (b - a);

            for (size_t j = 0; j < targets.size(); j++) {
                glm::vec3 closest;
                if (const float distance = distance_to_segment(a, b, targets[j], closest);
                    distance < min_distances[j]) {
                    min_distances[j] = distance;
                    closest_points[j] = closest;
                }
            }

            if (hit.has_value()) break;
        }

        float best = 0.f;
        for (size_t j = 0; j < targets.size(); j++)
            best = std::max(best, scorer.score(fire_pos, targets[j], closest_points[j]));

        return best;
    }

}// namespace arenai::model
