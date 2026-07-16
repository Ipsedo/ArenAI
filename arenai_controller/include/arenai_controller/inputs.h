//
// Created by samuel on 19/03/2023.
//

#ifndef ARENAI_INPUTS_H
#define ARENAI_INPUTS_H

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

    struct user_input {
        joystick left_joystick;
        joystick right_joystick;

        button fire_button;
    };

}// namespace arenai::controller

#endif// ARENAI_INPUTS_H
