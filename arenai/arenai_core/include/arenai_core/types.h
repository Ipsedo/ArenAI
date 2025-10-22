//
// Created by samuel on 06/10/2025.
//

#ifndef ARENAI_TYPES_H
#define ARENAI_TYPES_H

#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_view/pbuffer_renderer.h>

struct State {
    image<uint8_t> vision;
    std::vector<float> proprioception;
};

typedef float Reward;

typedef bool IsFinish;

typedef user_input Action;

#endif//ARENAI_TYPES_H
