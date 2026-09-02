//
// Created by samuel on 27/08/2026.
//

#include <utility>

#include <arenai_controller/input_names.h>

namespace arenai::controller {

    namespace {

        constexpr std::pair<Key, const char *> KEY_NAMES[] = {
            // clang-format off
            {Key::A, "A"}, {Key::B, "B"}, {Key::C, "C"}, {Key::D, "D"}, {Key::E, "E"},
            {Key::F, "F"}, {Key::G, "G"}, {Key::H, "H"}, {Key::I, "I"}, {Key::J, "J"},
            {Key::K, "K"}, {Key::L, "L"}, {Key::M, "M"}, {Key::N, "N"}, {Key::O, "O"},
            {Key::P, "P"}, {Key::Q, "Q"}, {Key::R, "R"}, {Key::S, "S"}, {Key::T, "T"},
            {Key::U, "U"}, {Key::V, "V"}, {Key::W, "W"}, {Key::X, "X"}, {Key::Y, "Y"},
            {Key::Z, "Z"},
            {Key::Num0, "0"}, {Key::Num1, "1"}, {Key::Num2, "2"}, {Key::Num3, "3"},
            {Key::Num4, "4"}, {Key::Num5, "5"}, {Key::Num6, "6"}, {Key::Num7, "7"},
            {Key::Num8, "8"}, {Key::Num9, "9"},
            {Key::F1, "F1"}, {Key::F2, "F2"}, {Key::F3, "F3"}, {Key::F4, "F4"},
            {Key::F5, "F5"}, {Key::F6, "F6"}, {Key::F7, "F7"}, {Key::F8, "F8"},
            {Key::F9, "F9"}, {Key::F10, "F10"}, {Key::F11, "F11"}, {Key::F12, "F12"},
            {Key::Up, "Up"}, {Key::Down, "Down"}, {Key::Left, "Left"}, {Key::Right, "Right"},
            {Key::Space, "Space"}, {Key::Escape, "Escape"}, {Key::Enter, "Enter"},
            {Key::Tab, "Tab"}, {Key::Backspace, "Backspace"}, {Key::Insert, "Insert"},
            {Key::Delete, "Delete"}, {Key::Home, "Home"}, {Key::End, "End"},
            {Key::PageUp, "PageUp"}, {Key::PageDown, "PageDown"},
            {Key::LeftShift, "LeftShift"}, {Key::RightShift, "RightShift"},
            {Key::LeftControl, "LeftControl"}, {Key::RightControl, "RightControl"},
            {Key::LeftAlt, "LeftAlt"}, {Key::RightAlt, "RightAlt"},
            {Key::Comma, ","}, {Key::Period, "."}, {Key::Semicolon, ";"},
            {Key::Apostrophe, "'"}, {Key::Slash, "/"}, {Key::Backslash, "\\"},
            {Key::Minus, "-"}, {Key::Equal, "="}, {Key::LeftBracket, "["},
            {Key::RightBracket, "]"}, {Key::Grave, "`"},
            // clang-format on
        };

        constexpr std::pair<GamepadButton, const char *> GAMEPAD_BUTTON_NAMES[] = {
            {GamepadButton::A, "A"},
            {GamepadButton::B, "B"},
            {GamepadButton::X, "X"},
            {GamepadButton::Y, "Y"},
            {GamepadButton::RB, "RB"},
            {GamepadButton::LB, "LB"},
            {GamepadButton::Start, "Start"},
            {GamepadButton::DPadUp, "DPadUp"},
            {GamepadButton::DPadDown, "DPadDown"},
            {GamepadButton::DPadLeft, "DPadLeft"},
            {GamepadButton::DPadRight, "DPadRight"},
        };

    }// namespace

    const char *to_string(const Key key) {
        for (const auto &[value, name]: KEY_NAMES)
            if (value == key) return name;
        return "";
    }

    Key key_from_string(const std::string_view name) {
        for (const auto &[value, key_name]: KEY_NAMES)
            if (name == key_name) return value;
        return Key::Unknown;
    }

    const char *to_string(const GamepadButton button) {
        for (const auto &[value, name]: GAMEPAD_BUTTON_NAMES)
            if (value == button) return name;
        return "";
    }

    std::optional<GamepadButton> gamepad_button_from_string(const std::string_view name) {
        for (const auto &[value, button_name]: GAMEPAD_BUTTON_NAMES)
            if (name == button_name) return value;
        return std::nullopt;
    }

}// namespace arenai::controller
