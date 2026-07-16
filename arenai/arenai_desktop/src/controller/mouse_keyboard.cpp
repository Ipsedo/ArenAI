//
// Created by samuel on 16/03/2026.
//

#include "./mouse_keyboard.h"

#include <cmath>
#include <utility>

namespace arenai::desktop {

    PlayerMouseKeyboardHandler::PlayerMouseKeyboardHandler(
        std::shared_ptr<view::AbstractWindow> window, const view::AbstractRenderer &renderer)
        : window(std::move(window)), renderer(renderer), last_mouse_x(0.), last_mouse_y(0.),
          current_dir(0.f), current_speed(0.f), current_turret_rotation(0.f),
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
        on_event({std::make_pair(key, action), std::nullopt, last_mouse_x, last_mouse_y});
    }

    void PlayerMouseKeyboardHandler::on_mouse_move(const double x, const double y) {
        last_mouse_x = x;
        last_mouse_y = y;
        on_event({std::nullopt, std::nullopt, x, y});
    }

    void PlayerMouseKeyboardHandler::on_mouse_button(
        const controller::MouseButton button, const controller::InputAction action) {
        on_event({std::nullopt, std::make_pair(button, action), last_mouse_x, last_mouse_y});
    }

    std::tuple<bool, controller::user_input>
    PlayerMouseKeyboardHandler::to_output(const PlayerMouseKeyboardInput event) {

        bool need_fire = false;

        // keys
        if (event.key) {
            const auto [key, action] = *event.key;

            if (action == controller::InputAction::Press) switch (key) {
                    case controller::Key::W: current_speed = 1.f; break;
                    case controller::Key::S: current_speed = -1.f; break;
                    case controller::Key::A: current_dir = -1.f; break;
                    case controller::Key::D: current_dir = 1.f; break;
                    case controller::Key::Space: need_fire = true; break;
                    case controller::Key::Escape: cursor_captured = false; break;
                    default: break;
                }

            if (action == controller::InputAction::Release) {
                if (key == controller::Key::W || key == controller::Key::S) current_speed = 0.f;
                if (key == controller::Key::A || key == controller::Key::D) current_dir = 0.f;
            }
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
            if (button == controller::MouseButton::Left
                && action == controller::InputAction::Press) {
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
