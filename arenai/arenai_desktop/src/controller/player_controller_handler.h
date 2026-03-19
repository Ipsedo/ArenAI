//
// Created by samuel on 16/03/2026.
//

#ifndef ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H
#define ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H

#include <GLFW/glfw3.h>

#include <arenai_controller/handler.h>

struct GlfwInput {
    int key;
    int key_action;

    float mouse_x;
    float mouse_y;

    int mouse_button;
    int mouse_button_action;
};

class MouseKeyboardPlayerControllerHandler : public ControllerHandler<GlfwInput> {
public:
    explicit MouseKeyboardPlayerControllerHandler(GLFWwindow *window);

protected:
    std::tuple<bool, user_input> to_output(GlfwInput event) override;

private:
    GLFWwindow *window;

    bool first_use;

    float current_dir;
    float current_speed;

    double last_mouse_x;
    double last_mouse_y;
};

#endif//ARENAI_DESKTOP_PLAYER_CONTROLLER_HANDLER_H
