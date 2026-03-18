//
// Created by samuel on 16/03/2026.
//

#include "./player_controller_handler.h"

MouseKeyboardPlayerControllerHandler::MouseKeyboardPlayerControllerHandler(GLFWwindow *window)
    : current_dir(0.f), current_speed(0.f), last_mouse_x(0.f), last_mouse_y(0.f) {
    glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
}

std::tuple<bool, user_input>
MouseKeyboardPlayerControllerHandler::to_output(const GlfwInput event) {

    bool need_fire = false;

    constexpr float speed_factor = 0.01f;
    constexpr float dir_factor = 0.05f;

    if (event.key_action == GLFW_REPEAT || event.key_action == GLFW_PRESS) switch (event.key) {
            case GLFW_KEY_W: current_speed += speed_factor; break;
            case GLFW_KEY_S: current_speed -= speed_factor; break;
            case GLFW_KEY_A: current_dir -= dir_factor; break;
            case GLFW_KEY_D: current_dir += dir_factor; break;
            case GLFW_KEY_SPACE: need_fire = true; break;
            default: break;
        }

    if (event.key_action == GLFW_RELEASE) {
        if (event.key == GLFW_KEY_W || event.key == GLFW_KEY_S) current_speed = 0.f;
        if (event.key == GLFW_KEY_Z || event.key == GLFW_KEY_D) current_dir = 0.f;
    }

    constexpr float factor = 0.001f;

    const float turret_rot = (event.mouse_x - last_mouse_x) * factor;
    last_mouse_x = event.mouse_x;

    const float canon_rot = (event.mouse_y - last_mouse_y) * factor;
    last_mouse_y = event.mouse_y;

    return {true, {{current_dir, current_speed}, {turret_rot, canon_rot}, {need_fire}}};
}
