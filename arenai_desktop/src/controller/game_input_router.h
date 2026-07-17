//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_DESKTOP_GAME_INPUT_ROUTER_H
#define ARENAI_DESKTOP_GAME_INPUT_ROUTER_H

#include <functional>
#include <memory>

#include <arenai_controller/callback.h>

namespace arenai::desktop {

    // Single owner of the window's input slots during a game: forwards events
    // to the game handlers or to the pause menu depending on the pause state,
    // and turns Escape / gamepad Start into a pause-toggle request. Routing is
    // an application policy: neither the environment nor the gui knows the
    // other exists.
    class GameInputRouter final : public controller::AbstractKeyboardCallback,
                                  public controller::AbstractGamepadCallback {
    public:
        GameInputRouter(
            std::shared_ptr<controller::AbstractKeyboardCallback> game_keyboard,
            std::shared_ptr<controller::AbstractGamepadCallback> game_gamepad,
            std::shared_ptr<controller::AbstractKeyboardCallback> pause_input,
            std::function<void()> on_pause_toggle);

        void set_paused(bool paused);

        // keyboard
        void on_key(controller::Key key, controller::InputAction action) override;
        void on_mouse_move(double x, double y) override;
        void
        on_mouse_button(controller::MouseButton button, controller::InputAction action) override;
        void on_scroll(double x_offset, double y_offset) override;

        // gamepad
        void on_gamepad_button(
            controller::GamepadButton button, controller::InputAction action) override;
        void on_joystick(double x, double y, controller::GamepadJoystick stick) override;
        void on_trigger(double z, controller::GamepadTrigger trigger) override;

    private:
        const std::shared_ptr<controller::AbstractKeyboardCallback> &keyboard_sink() const;

        std::shared_ptr<controller::AbstractKeyboardCallback> game_keyboard_;
        std::shared_ptr<controller::AbstractGamepadCallback> game_gamepad_;
        std::shared_ptr<controller::AbstractKeyboardCallback> pause_input_;
        std::function<void()> on_pause_toggle_;

        bool paused_ = false;
    };

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_GAME_INPUT_ROUTER_H
