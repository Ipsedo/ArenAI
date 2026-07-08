//
// Created by samuel on 16/03/2026.
//

#include "./player_controller_handler.h"

#include <cmath>
#include <utility>

namespace arenai::desktop {

    MouseKeyboardPlayerControllerHandler::MouseKeyboardPlayerControllerHandler(
        std::shared_ptr<view::AbstractWindow> window)
        : window(std::move(window)), last_mouse_x(0.), last_mouse_y(0.), current_dir(0.f),
          current_speed(0.f), current_turret_rotation(0.f), current_canon_rotation(0.f),
          cursor_captured(true) {

        const auto [window_width, window_height] = this->window->size();
        const auto center_x = static_cast<double>(window_width) / 2.,
                   center_y = static_cast<double>(window_height) / 2.;

        last_mouse_x = center_x;
        last_mouse_y = center_y;

        this->window->set_cursor_mode(view::CursorMode::Disabled);
        this->window->set_cursor_position(center_x, center_y);
    }

    void MouseKeyboardPlayerControllerHandler::on_key(
        const view::Key key, const view::InputAction action) {
        on_event({std::make_pair(key, action), std::nullopt, last_mouse_x, last_mouse_y});
    }

    void MouseKeyboardPlayerControllerHandler::on_mouse_move(const double x, const double y) {
        last_mouse_x = x;
        last_mouse_y = y;
        on_event({std::nullopt, std::nullopt, x, y});
    }

    void MouseKeyboardPlayerControllerHandler::on_mouse_button(
        const view::MouseButton button, const view::InputAction action) {
        on_event({std::nullopt, std::make_pair(button, action), last_mouse_x, last_mouse_y});
    }

    std::tuple<bool, controller::user_input>
    MouseKeyboardPlayerControllerHandler::to_output(const PlayerRawInput event) {

        bool need_fire = false;

        // keys
        if (event.key) {
            const auto [key, action] = *event.key;

            if (action == view::InputAction::Press) switch (key) {
                    case view::Key::W: current_speed = 1.f; break;
                    case view::Key::S: current_speed = -1.f; break;
                    case view::Key::A: current_dir = -1.f; break;
                    case view::Key::D: current_dir = 1.f; break;
                    case view::Key::Space: need_fire = true; break;
                    case view::Key::Escape: cursor_captured = false; break;
                    default: break;
                }

            if (action == view::InputAction::Release) {
                if (key == view::Key::W || key == view::Key::S) current_speed = 0.f;
                if (key == view::Key::A || key == view::Key::D) current_dir = 0.f;
            }
        }

        // mouse
        const auto [window_width, window_height] = window->size();
        const auto center_x = static_cast<double>(window_width) / 2.,
                   center_y = static_cast<double>(window_height) / 2.;

        if (cursor_captured) {
            window->set_cursor_mode(view::CursorMode::Disabled);

            // controllers consume rad/frame deltas, so the normalized mouse
            // displacement is scaled into radians here.
            constexpr float factor = 0.4f * static_cast<float>(M_PI);

            current_turret_rotation =
                factor * static_cast<float>((event.mouse_x - center_x) / center_x);
            current_canon_rotation =
                factor * static_cast<float>((event.mouse_y - center_y) / center_y);

            window->set_cursor_position(center_x, center_y);
        } else {
            window->set_cursor_mode(view::CursorMode::Normal);

            current_turret_rotation = 0.f;
            current_canon_rotation = 0.f;
        }

        // mouse buttons
        if (event.button) {
            const auto [button, action] = *event.button;
            if (button == view::MouseButton::Left && action == view::InputAction::Press) {
                need_fire = true;
                cursor_captured = true;
            }
        }

        return {
            true,
            {{current_dir, current_speed},
             {current_turret_rotation, current_canon_rotation},
             {need_fire}}};
    }

}// namespace arenai::desktop
