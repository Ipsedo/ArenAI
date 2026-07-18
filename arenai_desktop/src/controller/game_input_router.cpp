//
// Created by samuel on 17/07/2026.
//

#include "./game_input_router.h"

#include <utility>

namespace arenai::desktop {

    GameInputRouter::GameInputRouter(
        std::shared_ptr<controller::AbstractKeyboardCallback> game_keyboard,
        std::shared_ptr<controller::AbstractGamepadCallback> game_gamepad,
        std::shared_ptr<controller::AbstractKeyboardCallback> pause_input,
        std::shared_ptr<controller::AbstractGamepadCallback> pause_gamepad_input,
        std::function<void()> on_pause_toggle)
        : game_keyboard_(std::move(game_keyboard)), game_gamepad_(std::move(game_gamepad)),
          pause_input_(std::move(pause_input)),
          pause_gamepad_input_(std::move(pause_gamepad_input)),
          on_pause_toggle_(std::move(on_pause_toggle)) {}

    void GameInputRouter::set_paused(const bool paused) {
        paused_ = paused;

        // keys held when the popup opens would otherwise stay "pressed" for
        // the whole pause (their release goes to the menu): zero the movement
        // state with synthetic releases
        if (paused_ && game_keyboard_)
            for (const auto key:
                 {controller::Key::W, controller::Key::A, controller::Key::S, controller::Key::D})
                game_keyboard_->on_key(key, controller::InputAction::Release);
    }

    const std::shared_ptr<controller::AbstractKeyboardCallback> &
    GameInputRouter::keyboard_sink() const {
        return paused_ ? pause_input_ : game_keyboard_;
    }

    const std::shared_ptr<controller::AbstractGamepadCallback> &
    GameInputRouter::gamepad_sink() const {
        return paused_ ? pause_gamepad_input_ : game_gamepad_;
    }

    void GameInputRouter::on_key(const controller::Key key, const controller::InputAction action) {
        if (key == controller::Key::Escape && action == controller::InputAction::Press) {
            on_pause_toggle_();
            return;
        }

        if (const auto &sink = keyboard_sink()) sink->on_key(key, action);
    }

    void GameInputRouter::on_mouse_move(const double x, const double y) {
        if (const auto &sink = keyboard_sink()) sink->on_mouse_move(x, y);
    }

    void GameInputRouter::on_mouse_button(
        const controller::MouseButton button, const controller::InputAction action) {
        if (const auto &sink = keyboard_sink()) sink->on_mouse_button(button, action);
    }

    void GameInputRouter::on_scroll(const double x_offset, const double y_offset) {
        if (const auto &sink = keyboard_sink()) sink->on_scroll(x_offset, y_offset);
    }

    void GameInputRouter::on_gamepad_button(
        const controller::GamepadButton button, const controller::InputAction action) {
        if (button == controller::GamepadButton::Start
            && action == controller::InputAction::Press) {
            on_pause_toggle_();
            return;
        }

        if (const auto &sink = gamepad_sink()) sink->on_gamepad_button(button, action);
    }

    void GameInputRouter::on_joystick(
        const double x, const double y, const controller::GamepadJoystick stick) {
        if (const auto &sink = gamepad_sink()) sink->on_joystick(x, y, stick);
    }

    void GameInputRouter::on_trigger(const double z, const controller::GamepadTrigger trigger) {
        if (const auto &sink = gamepad_sink()) sink->on_trigger(z, trigger);
    }

}// namespace arenai::desktop
