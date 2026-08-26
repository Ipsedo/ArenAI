//
// Created by samuel on 16/03/2026.
//

#include "./mouse_keyboard.h"

#include <utility>

namespace arenai::desktop {

    PlayerMouseKeyboardHandler::PlayerMouseKeyboardHandler(
        std::shared_ptr<view::AbstractWindow> window, const view::AbstractRenderer &renderer,
        const KeyboardBindings &bindings)
        : window(std::move(window)), renderer(renderer), bindings(bindings), last_mouse_x(0.),
          last_mouse_y(0.), current_dir(0.f), current_speed(0.f), current_turret_rotation(0.f),
          current_canon_rotation(0.f), cursor_captured(true) {

        const auto center_x = static_cast<double>(renderer.get_width()) / 2.,
                   center_y = static_cast<double>(renderer.get_height()) / 2.;

        last_mouse_x = center_x;
        last_mouse_y = center_y;

        this->window->set_cursor_mode(controller::CursorMode::Disabled);
        this->window->set_cursor_position(center_x, center_y);
    }

    void PlayerMouseKeyboardHandler::on_key(
        const controller::Key key, const controller::InputAction action) {
        on_event(
            {.key = std::make_pair(key, action),
             .button = std::nullopt,
             .mouse_x = last_mouse_x,
             .mouse_y = last_mouse_y});
    }

    void PlayerMouseKeyboardHandler::on_mouse_move(const double x, const double y) {
        last_mouse_x = x;
        last_mouse_y = y;
        on_event({.key = std::nullopt, .button = std::nullopt, .mouse_x = x, .mouse_y = y});
    }

    void PlayerMouseKeyboardHandler::on_mouse_button(
        const controller::MouseButton button, const controller::InputAction action) {
        on_event(
            {.key = std::nullopt,
             .button = std::make_pair(button, action),
             .mouse_x = last_mouse_x,
             .mouse_y = last_mouse_y});
    }

    void PlayerMouseKeyboardHandler::apply_binding(
        const KeyboardBinding &input, const controller::InputAction action, bool &need_fire) {
        if (action == controller::InputAction::Press) {
            if (input == bindings.forward) current_speed = 1.f;
            else if (input == bindings.backward) current_speed = -1.f;
            else if (input == bindings.turn_left) current_dir = -1.f;
            else if (input == bindings.turn_right) current_dir = 1.f;
            else if (input == bindings.fire) need_fire = true;
        } else if (action == controller::InputAction::Release) {
            if (input == bindings.forward || input == bindings.backward) current_speed = 0.f;
            if (input == bindings.turn_left || input == bindings.turn_right) current_dir = 0.f;
        }
    }

    std::tuple<bool, controller::user_input>
    PlayerMouseKeyboardHandler::to_output(const PlayerMouseKeyboardInput event) {

        bool need_fire = false;

        // keys
        if (event.key) {
            const auto [key, action] = *event.key;

            apply_binding(KeyboardBinding(key), action, need_fire);

            // Escape stays hardwired: it hands the cursor back to the OS
            if (key == controller::Key::Escape && action == controller::InputAction::Press)
                cursor_captured = false;
        }

        // mouse
        const auto center_x = static_cast<double>(renderer.get_width()) / 2.,
                   center_y = static_cast<double>(renderer.get_height()) / 2.;

        if (cursor_captured) {
            window->set_cursor_mode(controller::CursorMode::Disabled);

            // controllers consume rad/frame deltas, so the normalized mouse
            // displacement is scaled into radians here.
            constexpr float factor = 0.4f * static_cast<float>(M_PI);

            current_turret_rotation =
                factor * static_cast<float>((event.mouse_x - center_x) / center_x);
            current_canon_rotation =
                factor * static_cast<float>((event.mouse_y - center_y) / center_y);

            window->set_cursor_position(center_x, center_y);
        } else {
            window->set_cursor_mode(controller::CursorMode::Normal);

            current_turret_rotation = 0.f;
            current_canon_rotation = 0.f;
        }

        // mouse buttons
        if (event.button) {
            const auto [button, action] = *event.button;

            apply_binding(KeyboardBinding(button), action, need_fire);

            // a left click always recaptures the cursor after an Escape
            if (button == controller::MouseButton::Left && action == controller::InputAction::Press)
                cursor_captured = true;
        }

        return {
            true,
            {.left_joystick = {.x = current_dir, .y = current_speed},
             .right_joystick = {.x = current_turret_rotation, .y = current_canon_rotation},
             .fire_button = {need_fire}}};
    }

}// namespace arenai::desktop
