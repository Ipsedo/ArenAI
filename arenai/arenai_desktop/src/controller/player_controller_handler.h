//
// Created by samuel on 16/03/2026.
//

#ifndef ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H
#define ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H

#include <arenai_controller/handler.h>

struct GlfwInput {
    int key;
    int key_action;

    float mouse_x;
    float mouse_y;

    int mouse_button;
    int mouse_button_action;
};

class DesktopPlayerControllerHandler : public ControllerHandler<GlfwInput> {
public:
    DesktopPlayerControllerHandler();

protected:
    std::tuple<bool, user_input> to_output(GlfwInput event) override;
};

#endif//ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H
