//
// Created by samuel on 06/10/2025.
//

#ifndef PHYVR_TYPES_H
#define PHYVR_TYPES_H

#include <vector>

#include <phyvr_controller/inputs.h>
#include <phyvr_view/pbuffer_renderer.h>

#define ENEMY_VISION_SIZE 128
// 3 * (pos + vel + acc) + 3 * (ang + vel_ang + acc_and)
#define ENEMY_PROPRIOCEPTION_SIZE ((3 * 2 + 4 + 3) * (6 + 3))
#define ENEMY_NB_ACTION (2 + 2 + 1)

struct State {
    image<uint8_t> vision;
    std::vector<float> proprioception;
};

typedef float Reward;

typedef bool IsFinish;

typedef user_input Action;

#endif//PHYVR_TYPES_H
