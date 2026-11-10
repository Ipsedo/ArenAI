#ifndef ARENAI_DESKTOP_GAME_INPUT_ROUTER_H
#define ARENAI_DESKTOP_GAME_INPUT_ROUTER_H

#include <functional>
#include <memory>

#include <arenai_controller/callback.h>

#include "./bindings.h"

namespace arenai::desktop {

    class GameInputRouter final : public controller::AbstractKeyboardCallback,
                                  public controller::AbstractGamepadCallback {
    public:
        GameInputRouter(
            std::shared_ptr<AbstractKeyboardCallback> game_keyboard,
            std::shared_ptr<AbstractGamepadCallback> game_gamepad,
            std::shared_ptr<AbstractKeyboardCallback> pause_input,
            std::shared_ptr<AbstractGamepadCallback> pause_gamepad_input,
            std::function<void()> on_pause_toggle, const KeyboardBindings &keyboard_bindings);

        void set_paused(bool paused);

        void on_key(controller::Key key, controller::InputAction action) override;
        void on_mouse_move(double x, double y) override;
        void
        on_mouse_button(controller::MouseButton button, controller::InputAction action) override;
        void on_scroll(double x_offset, double y_offset) override;

        void on_gamepad_button(
            controller::GamepadButton button, controller::InputAction action) override;
        void on_joystick(double x, double y, controller::GamepadJoystick stick) override;
        void on_trigger(double z, controller::GamepadTrigger trigger) override;

    private:
        const std::shared_ptr<AbstractKeyboardCallback> &keyboard_sink() const;
        const std::shared_ptr<AbstractGamepadCallback> &gamepad_sink() const;

        std::shared_ptr<AbstractKeyboardCallback> game_keyboard_;
        std::shared_ptr<AbstractGamepadCallback> game_gamepad_;
        std::shared_ptr<AbstractKeyboardCallback> pause_input_;
        std::shared_ptr<AbstractGamepadCallback> pause_gamepad_input_;
        std::function<void()> on_pause_toggle_;
        KeyboardBindings keyboard_bindings_;

        bool paused_ = false;
    };

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_GAME_INPUT_ROUTER_H
