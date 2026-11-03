//
// Created by claude on 03/11/2026.
//

#include "./aim_scorer.h"

namespace arenai::model {

    AimScorer::AimScorer(const float tolerance_min, const float tolerance_angle)
        : tolerance_min(tolerance_min), tolerance_angle(tolerance_angle) {}

    float AimScorer::tolerance(const float shot_distance) const {
        return tolerance_min + tolerance_angle * shot_distance;
    }

    float AimScorer::score(
        const glm::vec3 &fire_pos, const glm::vec3 &enemy_pos, const glm::vec3 &shell_pos) const {
        const float shot_distance = glm::length(enemy_pos - fire_pos);
        const float miss_distance = glm::length(shell_pos - enemy_pos);

        const float ratio = miss_distance / tolerance(shot_distance);

        return 1.f / (1.f + ratio * ratio);
    }

}// namespace arenai::model
