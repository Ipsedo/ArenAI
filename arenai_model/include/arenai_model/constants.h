//
// Created by samuel on 09/06/2026.
//

#ifndef ARENAI_MODEL_CONSTANTS_H
#define ARENAI_MODEL_CONSTANTS_H

namespace arenai::model {

    constexpr float WHEEL_RADIAL_VELOCITY = std::numbers::pi * 5.f;

    // (pos, vel, forward, up, ang_vel) * (9 items: 6 wheels, 1 chassis, 1 turret, 1 canon) - chassis pos + remaining shells ratio
    constexpr int ENEMY_PROPRIOCEPTION_SIZE = (3 + 3 + 3 + 3 + 3) * (6 + 3) - 3 + 1;
    constexpr int ENEMY_NB_CONTINUOUS_ACTION = 2 + 2;
    constexpr int ENEMY_NB_DISCRETE_ACTION = 2;

    constexpr float ENEMY_TURRET_RADIAL_VELOCITY = std::numbers::pi * 1.f;

}// namespace arenai::model

#endif//ARENAI_MODEL_CONSTANTS_H
