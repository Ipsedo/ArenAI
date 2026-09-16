//
// Created by samuel on 06/10/2025.
//

#ifndef ARENAI_TYPES_H
#define ARENAI_TYPES_H

#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_view/renderer.h>

namespace arenai::core {
    struct State {
        view::image<uint8_t> vision;
        std::vector<float> proprioception;
    };

    typedef float Reward;

    typedef bool IsDone;

    // done from starving out: the episode ends for the tank, but the value target must
    // bootstrap instead of cutting the return (truncation, not termination)
    typedef bool IsTruncated;

    typedef controller::user_input Action;
}// namespace arenai::core

#endif//ARENAI_TYPES_H
