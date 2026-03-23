//
// Created by samuel on 16/03/2026.
//

#include "./player_controller_handler.h"

#include <algorithm>

MouseKeyboardPlayerControllerHandler::MouseKeyboardPlayerControllerHandler(GLFWwindow *window)
    : window(window), current_dir(0.f), current_speed(0.f), current_turret_rotation(0.f),
      current_canon_rotation(0.f), cursor_captured(true) {

    int window_width = 0, window_height = 0;
    glfwGetWindowSize(window, &window_width, &window_height);
    const float center_x = static_cast<float>(window_width) / 2.f,
                center_y = static_cast<float>(window_height) / 2.f;

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPos(window, center_x, center_y);
}

std::tuple<bool, user_input>
MouseKeyboardPlayerControllerHandler::to_output(const GlfwInput event) {

    // keys
    bool need_fire = false;

    if (event.key_action == GLFW_PRESS) switch (event.key) {
            case GLFW_KEY_W: current_speed = 1.f; break;
            case GLFW_KEY_S: current_speed = -1.f; break;
            case GLFW_KEY_A: current_dir = -1.f; break;
            case GLFW_KEY_D: current_dir = 1.f; break;
            case GLFW_KEY_SPACE: need_fire = true; break;
            case GLFW_KEY_ESCAPE: cursor_captured = false; break;
            default: break;
        }

    if (event.key_action == GLFW_RELEASE) {
        if (event.key == GLFW_KEY_W || event.key == GLFW_KEY_S) current_speed = 0.f;
        if (event.key == GLFW_KEY_A || event.key == GLFW_KEY_D) current_dir = 0.f;
    }

    // mouse
    int window_width = 0, window_height = 0;
    glfwGetWindowSize(window, &window_width, &window_height);
    const float center_x = static_cast<float>(window_width) / 2.f,
                center_y = static_cast<float>(window_height) / 2.f;

    if (cursor_captured) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        constexpr float factor = 0.4f;

        current_turret_rotation = factor * (event.mouse_x - center_x) / center_x;
        current_canon_rotation = factor * (event.mouse_y - center_y) / center_y;

        glfwSetCursorPos(window, center_x, center_y);
    } else {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        current_turret_rotation = 0.f;
        current_canon_rotation = 0.f;
    }

    // mouse buttons
    if (event.mouse_button == GLFW_MOUSE_BUTTON_LEFT && event.mouse_button_action == GLFW_PRESS) {
        need_fire = true;
        cursor_captured = true;
    }

    return {
        true,
        {{current_dir, current_speed},
         {current_turret_rotation, current_canon_rotation},
         {need_fire}}};
}
