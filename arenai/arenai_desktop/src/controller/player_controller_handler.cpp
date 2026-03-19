//
// Created by samuel on 16/03/2026.
//

#include "./player_controller_handler.h"

#include <algorithm>

MouseKeyboardPlayerControllerHandler::MouseKeyboardPlayerControllerHandler(GLFWwindow *window)
    : window(window), first_use(true), current_dir(0.f), current_speed(0.f), last_mouse_x(0.f),
      last_mouse_y(0.f) {
    glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
}

std::tuple<bool, user_input>
MouseKeyboardPlayerControllerHandler::to_output(const GlfwInput event) {
    if (first_use) {
        glfwGetCursorPos(window, &last_mouse_x, &last_mouse_y);
        first_use = false;
    }

    bool need_fire = false;

    if (event.key_action == GLFW_PRESS) switch (event.key) {
            case GLFW_KEY_W: current_speed = 1.f; break;
            case GLFW_KEY_S: current_speed = -1.f; break;
            case GLFW_KEY_A: current_dir = -1.f; break;
            case GLFW_KEY_D: current_dir = 1.f; break;
            case GLFW_KEY_SPACE: need_fire = true; break;
            default: break;
        }

    if (event.key_action == GLFW_RELEASE) {
        if (event.key == GLFW_KEY_W || event.key == GLFW_KEY_S) current_speed = 0.f;
        if (event.key == GLFW_KEY_A || event.key == GLFW_KEY_D) current_dir = 0.f;
    }

    constexpr float factor = 0.001f;

    const float turret_rot =
        std::clamp((event.mouse_x - static_cast<float>(last_mouse_x)) * factor, -1.f, 1.f);
    last_mouse_x = event.mouse_x;

    const float canon_rot =
        std::clamp((event.mouse_y - static_cast<float>(last_mouse_y)) * factor, -1.f, 1.f);
    last_mouse_y = event.mouse_y;

    if (event.mouse_button == GLFW_MOUSE_BUTTON_LEFT && event.mouse_button_action == GLFW_PRESS)
        need_fire = true;

    return {true, {{current_dir, current_speed}, {turret_rot, canon_rot}, {need_fire}}};
}
