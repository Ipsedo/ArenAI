//
// Created by claude on 03/11/2026.
//

#ifndef ARENAI_AIM_SCORER_H
#define ARENAI_AIM_SCORER_H

#include <glm/glm.hpp>

namespace arenai::model {

    class AimScorer {
    public:
        AimScorer(float tolerance_min, float tolerance_angle);

        float tolerance(float shot_distance) const;

        float score(
            const glm::vec3 &fire_pos, const glm::vec3 &enemy_pos,
            const glm::vec3 &shell_pos) const;

    private:
        float tolerance_min;
        float tolerance_angle;
    };

}// namespace arenai::model

#endif// ARENAI_AIM_SCORER_H
