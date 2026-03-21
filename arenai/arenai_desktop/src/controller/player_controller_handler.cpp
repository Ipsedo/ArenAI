//
// Created by samuel on 16/03/2026.
//

#include "./player_controller_handler.h"

#include <algorithm>

MouseKeyboardPlayerControllerHandler::MouseKeyboardPlayerControllerHandler(GLFWwindow *window)
    : window(window), current_dir(0.f), current_speed(0.f), current_turret_rotation(0.F),
      current_canon_rotation(0.f) {}

std::tuple<bool, user_input>
MouseKeyboardPlayerControllerHandler::to_output(const GlfwInput event) {

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

    constexpr float factor = 2e-2f;

    int window_width = 0, window_height = 0;
    glfwGetWindowSize(window, &window_width, &window_height);

    const float center_x = static_cast<float>(window_width) / 2.f,
                center_y = static_cast<float>(window_height) / 2.f;

    current_turret_rotation =
        (event.mouse_x - center_x) / static_cast<float>(window_width) * factor;
    current_canon_rotation =
        (event.mouse_y - center_y) / static_cast<float>(window_height) * factor;

    if (event.mouse_button == GLFW_MOUSE_BUTTON_LEFT && event.mouse_button_action == GLFW_PRESS)
        need_fire = true;

    return {
        true,
        {{current_dir, current_speed},
         {current_turret_rotation, current_canon_rotation},
         {need_fire}}};
}
