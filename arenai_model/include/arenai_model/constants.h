//
// Created by samuel on 09/06/2026.
//

#ifndef ARENAI_MODEL_CONSTANTS_H
#define ARENAI_MODEL_CONSTANTS_H

#include <cmath>

namespace arenai::model {

    constexpr float WHEEL_RADIAL_VELOCITY = static_cast<float>(M_PI) * 5.f;

    constexpr int ENEMY_PROPRIOCEPTION_SIZE = (3 + 3 + 3 + 3 + 3) * (6 + 3) - 3;
    constexpr int ENEMY_NB_CONTINUOUS_ACTION = 2 + 2;
    constexpr int ENEMY_NB_DISCRETE_ACTION = 2;

    constexpr float ENEMY_TURRET_RADIAL_VELOCITY = static_cast<float>(M_PI) * 1.f;

}// namespace arenai::model

#endif//ARENAI_MODEL_CONSTANTS_H
