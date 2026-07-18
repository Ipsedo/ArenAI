//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_CALLBACK_H
#define ARENAI_CALLBACK_H

namespace arenai::controller {
    enum class InputAction { Press, Release, Repeat };

    // keyboard
    enum class Key { Unknown, W, A, S, D, Space, Escape };
    enum class MouseButton { Left, Right, Middle };
    enum class CursorMode { Normal, Disabled };

    // gamepad
    enum class GamepadJoystick { Right, Left };
    enum class GamepadTrigger { Right, Left };
    enum class GamepadButton { A, B, X, Y, RB, LB, Start };

    class AbstractKeyboardCallback {
    public:
        virtual ~AbstractKeyboardCallback() = default;

        // keyboard
        virtual void on_key(Key key, InputAction action) = 0;
        virtual void on_mouse_move(double x, double y) = 0;
        virtual void on_mouse_button(MouseButton button, InputAction action) = 0;
        // no-op default: most handlers (tank controls) have no use for the wheel
        virtual void on_scroll(double x_offset, double y_offset) {}
    };

    class AbstractGamepadCallback {
    public:
        virtual ~AbstractGamepadCallback() = default;

        // gamepad
        virtual void on_gamepad_button(GamepadButton button, InputAction action) = 0;
        virtual void on_joystick(double x, double y, GamepadJoystick stick) = 0;
        virtual void on_trigger(double z, GamepadTrigger trigger) = 0;
    };
}// namespace arenai::controller

#endif//ARENAI_CALLBACK_H
