//
// Created by samuel on 09/06/2026.
//

#ifndef ARENAI_MODEL_CONSTANTS_H
#define ARENAI_MODEL_CONSTANTS_H

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

    // the player aims with the mouse, whose sensitivity was always a per-frame gain:
    // 0.4 pi per frame on the turret, expressed here as the slew speed it means at 30 Hz
    constexpr float PLAYER_TURRET_RADIAL_VELOCITY = std::numbers::pi * 12.f;
    constexpr float PLAYER_CANON_RADIAL_VELOCITY = 0.4f * PLAYER_TURRET_RADIAL_VELOCITY;

    constexpr float CANON_AIM_DISTANCE = 100.f;

    constexpr float ZOOM_MAGNIFICATION = 2.f;
    constexpr float ZOOM_TRANSITION_SECONDS = 0.25f;

}// namespace arenai::model

#endif//ARENAI_MODEL_CONSTANTS_H
