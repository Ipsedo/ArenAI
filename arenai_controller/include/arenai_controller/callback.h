//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_CALLBACK_H
#define ARENAI_CALLBACK_H

namespace arenai::controller {
    enum class InputAction { Press, Release, Repeat };

    // keyboard — the ranges A..Z, Num0..Num9 and F1..F12 are contiguous so
    // backends can convert them arithmetically
    enum class Key {
        Unknown,
        // clang-format off
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
        // clang-format on
        Up,
        Down,
        Left,
        Right,
        Space,
        Escape,
        Enter,
        Tab,
        Backspace,
        Insert,
        Delete,
        Home,
        End,
        PageUp,
        PageDown,
        LeftShift,
        RightShift,
        LeftControl,
        RightControl,
        LeftAlt,
        RightAlt,
        Comma,
        Period,
        Semicolon,
        Apostrophe,
        Slash,
        Backslash,
        Minus,
        Equal,
        LeftBracket,
        RightBracket,
        Grave,
    };
    enum class MouseButton { Left, Right, Middle };
    enum class CursorMode { Normal, Disabled };

    // gamepad
    enum class GamepadJoystick { Right, Left };
    enum class GamepadTrigger { Right, Left };
    enum class GamepadButton { A, B, X, Y, RB, LB, Start, DPadUp, DPadDown, DPadLeft, DPadRight };

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
