//
// Created by samuel on 06/10/2025.
//

#ifndef ARENAI_TYPES_H
#define ARENAI_TYPES_H

#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_view/pbuffer_renderer.h>

#define ENEMY_VISION_SIZE 128
// 3 * (pos + vel + acc) + 3 * (ang + vel_ang + acc_and)
#define ENEMY_PROPRIOCEPTION_SIZE ((3 * 2 + 4 + 3) * (6 + 3))
#define ENEMY_NB_ACTION (2 + 2 + 1)

#define SIGMA_MIN 1e-3f
#define SIGMA_MAX 50.f
#define ALPHA_BETA_BOUND 5.f

struct State {
    image<uint8_t> vision;
    std::vector<float> proprioception;
};

typedef float Reward;

typedef bool IsFinish;

typedef user_input Action;

#endif//ARENAI_TYPES_H
