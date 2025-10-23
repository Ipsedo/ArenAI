//
// Created by samuel on 21/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_CONSTANTS_H
#define ARENAI_TRAIN_HOST_CONSTANTS_H

#define ENEMY_VISION_SIZE 256
// 3 * (pos + vel + acc) + 3 * (ang + vel_ang + acc_and)
#define ENEMY_PROPRIOCEPTION_SIZE ((3 * 2 + 4 + 3) * (6 + 3))
#define ENEMY_NB_ACTION (2 + 2 + 1)

#define SIGMA_MIN 1e-6f
#define SIGMA_MAX 1e6f
#define ALPHA_BETA_BOUND 5.f

#endif//ARENAI_TRAIN_HOST_CONSTANTS_H
