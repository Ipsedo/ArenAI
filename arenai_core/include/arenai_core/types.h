#ifndef ARENAI_TYPES_H
#define ARENAI_TYPES_H

#include <vector>

#include <arenai_controller/inputs.h>
#include <arenai_model/constants.h>
#include <arenai_view/renderer.h>

namespace arenai::core {
    // the agent's vision, in pixels
    constexpr int VISION_HEIGHT = 128;
    constexpr int VISION_WIDTH = 256;

    // the agent's panorama: the views side by side, left to right, in pixels
    constexpr int PANORAMA_VIEWS = model::PANORAMA_VIEWS;
    constexpr int PANORAMA_HEIGHT = 32;
    constexpr int PANORAMA_WIDTH = 512;

    static_assert(
        PANORAMA_WIDTH
        == PANORAMA_VIEWS * PANORAMA_HEIGHT * static_cast<int>(model::PANORAMA_VIEW_ASPECT));

    struct State {
        view::image<uint8_t> vision;
        view::image<uint8_t> panorama;
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
