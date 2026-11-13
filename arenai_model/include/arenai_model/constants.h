#ifndef ARENAI_MODEL_CONSTANTS_H
#define ARENAI_MODEL_CONSTANTS_H

#include <numbers>

namespace arenai::model {

    constexpr float WHEEL_RADIAL_VELOCITY = std::numbers::pi * 5.f;

    // body proprioception: (pos, vel, forward, up, ang_vel, life_ratio) * (9 items: 6 wheels, 1 chassis, 1 turret, 1 canon) - chassis pos
    constexpr int ENEMY_PROPRIOCEPTION_SIZE = (3 + 3 + 3 + 3 + 3 + 1) * (6 + 3) - 3;
    // the network's state vector: the proprioception followed by the remaining shells ratio and the cooldown ratio
    constexpr int ENEMY_STATE_VECTOR_SIZE = ENEMY_PROPRIOCEPTION_SIZE + 2;
    constexpr int ENEMY_NB_CONTINUOUS_ACTION = 2 + 2;
    constexpr int ENEMY_NB_DISCRETE_ACTION = 2;

    constexpr float ENEMY_TURRET_RADIAL_VELOCITY = std::numbers::pi * 1.f;
    // the canon travels a much shorter range than the turret: it aims 2.5 times slower
    constexpr float ENEMY_CANON_RADIAL_VELOCITY = 0.4f * ENEMY_TURRET_RADIAL_VELOCITY;

    constexpr float PLAYER_TURRET_RADIAL_VELOCITY = std::numbers::pi * 12.f;
    constexpr float PLAYER_CANON_RADIAL_VELOCITY = 0.4f * PLAYER_TURRET_RADIAL_VELOCITY;

    // Hz, critically damped spring of the turret and canon hinge motors
    constexpr float AIM_HINGE_FREQUENCY = 20.f;

    constexpr float CANON_AIM_DISTANCE = 100.f;

    // m, the hull radius: a predicted pass within it scores the full aim potential
    constexpr float AIM_TARGET_RADIUS = 3.f;

    constexpr float GRAVITY = 9.8f;
    constexpr float SHELL_SPEED = 250.f;

    constexpr float DAMAGE_REWARD = 1.f;
    constexpr float KILL_REWARD = 1.f;
    constexpr float DEATH_PENALTY = 1.f;

    constexpr float ZOOM_MAGNIFICATION = 2.f;
    constexpr float ZOOM_TRANSITION_SECONDS = 0.25f;

}// namespace arenai::model

#endif//ARENAI_MODEL_CONSTANTS_H
