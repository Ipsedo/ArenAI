#ifndef ARENAI_MODEL_CONSTANTS_H
#define ARENAI_MODEL_CONSTANTS_H

#include <numbers>

namespace arenai::model {

    constexpr float WHEEL_RADIAL_VELOCITY = std::numbers::pi * 5.f;

    constexpr int ENEMY_PROPRIOCEPTION_SIZE = (3 + 3 + 3 + 3 + 3 + 1) * (6 + 3) - 3;
    constexpr int ENEMY_STATE_VECTOR_SIZE = ENEMY_PROPRIOCEPTION_SIZE + 2;
    constexpr int ENEMY_NB_CONTINUOUS_ACTION = 2 + 2;
    constexpr int ENEMY_NB_DISCRETE_ACTION = 2;

    constexpr float ENEMY_TURRET_RADIAL_VELOCITY = std::numbers::pi * 1.f;
    constexpr float ENEMY_CANON_RADIAL_VELOCITY = 0.4f * ENEMY_TURRET_RADIAL_VELOCITY;

    constexpr float PLAYER_TURRET_RADIAL_VELOCITY = std::numbers::pi * 12.f;
    constexpr float PLAYER_CANON_RADIAL_VELOCITY = 0.4f * PLAYER_TURRET_RADIAL_VELOCITY;

    constexpr float CANON_AIM_DISTANCE = 100.f;

    constexpr float GRAVITY = 9.8f;
    constexpr float SHELL_SPEED = 250.f;

    constexpr float DAMAGE_REWARD = 1.f;
    constexpr float KILL_REWARD = 1.f;
    constexpr float DEATH_PENALTY = 1.f;

    constexpr float ZOOM_MAGNIFICATION = 2.f;
    constexpr float ZOOM_TRANSITION_SECONDS = 0.25f;

}// namespace arenai::model

#endif//ARENAI_MODEL_CONSTANTS_H
