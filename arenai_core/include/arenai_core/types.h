#ifndef ARENAI_TYPES_H
#define ARENAI_TYPES_H

#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_view/renderer.h>

namespace arenai::core {
    struct State {
        view::image<uint8_t> vision;
        std::vector<float> proprioception;

        float remaining_shells_ratio;
        float cooldown_ratio;

        bool can_fire;
    };

    typedef float Reward;

    typedef bool IsDone;

    typedef bool IsTruncated;

    typedef controller::user_input Action;

    inline std::vector<Action> release_fire(std::vector<Action> actions) {
        for (auto &action: actions) action.fire_button.pressed = false;
        return actions;
    }
}// namespace arenai::core

#endif//ARENAI_TYPES_H
