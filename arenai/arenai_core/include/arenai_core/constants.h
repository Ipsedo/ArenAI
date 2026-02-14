//
// Created by samuel on 21/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_CONSTANTS_H
#define ARENAI_TRAIN_HOST_CONSTANTS_H

#define ENEMY_VISION_HEIGHT 256
#define ENEMY_VISION_WIDTH 512

// (position + velocity + forward + up + angle_velocity) * (6 * wheel + chassis + turret + canon) - chassis_pos
#define ENEMY_PROPRIOCEPTION_SIZE ((3 + 3 + 3 + 3 + 3) * (6 + 3) - 3)
#define ENEMY_NB_ACTION (2 + 2 + 1)

#define SIGMA_MIN 1e-8f
#define SIGMA_MAX 1e8f
#define EPSILON 1e-8f

#endif//ARENAI_TRAIN_HOST_CONSTANTS_H
