//
// Created by samuel on 09/06/2026.
//

#ifndef ARENAI_MODEL_CONSTANTS_H
#define ARENAI_MODEL_CONSTANTS_H

#include <limits>
#include <numbers>

namespace arenai::model {

    constexpr float WHEEL_RADIAL_VELOCITY = std::numbers::pi * 5.f;

    // (pos, vel, forward, up, ang_vel) * (9 items: 6 wheels, 1 chassis, 1 turret, 1 canon) - chassis pos + remaining shells ratio + cooldown ratio
    constexpr int ENEMY_PROPRIOCEPTION_SIZE = (3 + 3 + 3 + 3 + 3) * (6 + 3) - 3 + 1 + 1;
    constexpr int ENEMY_NB_CONTINUOUS_ACTION = 2 + 2;
    constexpr int ENEMY_NB_DISCRETE_ACTION = 2;

    constexpr float ENEMY_TURRET_RADIAL_VELOCITY = std::numbers::pi * 1.f;
    // the canon travels a much shorter range than the turret: it aims 2.5 times slower
    constexpr float ENEMY_CANON_RADIAL_VELOCITY = 0.4f * ENEMY_TURRET_RADIAL_VELOCITY;

    // a tank aiming at this speed reaches any target angle in a single frame: the player
    // aims with the mouse, which has no speed of its own to preserve
    constexpr float UNLIMITED_RADIAL_VELOCITY = std::numeric_limits<float>::infinity();

    constexpr float CANON_AIM_DISTANCE = 100.f;

    constexpr float ZOOM_MAGNIFICATION = 2.f;
    constexpr float ZOOM_TRANSITION_SECONDS = 0.25f;

}// namespace arenai::model

#endif//ARENAI_MODEL_CONSTANTS_H
