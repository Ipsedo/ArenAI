#ifndef ARENAI_INPUTS_H
#define ARENAI_INPUTS_H

#include <optional>

namespace arenai::controller {

    struct joystick {
        float x;
        float y;
    };

    struct slider {
        float level;
    };

    struct button {
        bool pressed;
    };

    // absolute aim set point: turret yaw as a (cos, sin) direction, canon pitch in [-1, 1]
    struct aim_target {
        float yaw_cos;
        float yaw_sin;
        float pitch;
    };

    struct user_input {
        joystick left_joystick;
        joystick right_joystick;

        button fire_button;
        button zoom_button;

        // when set, the turret and the canon reach it at their max rate, right_joystick is ignored
        std::optional<aim_target> aim;
    };

}// namespace arenai::controller

#endif// ARENAI_INPUTS_H
