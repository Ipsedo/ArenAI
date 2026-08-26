//
// Created by samuel on 27/08/2026.
//

#ifndef ARENAI_DESKTOP_CONTROLLER_BINDINGS_H
#define ARENAI_DESKTOP_CONTROLLER_BINDINGS_H

#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include <arenai_controller/callback.h>

namespace arenai::desktop {

    // A keyboard slot holds a key or a mouse button; nullopt = unbound (the
    // action lost its input to a conflicting reassignment and stays inert
    // until the player rebinds it).
    using KeyboardBinding = std::variant<controller::Key, controller::MouseButton>;

    struct KeyboardBindings {
        std::optional<KeyboardBinding> forward = controller::Key::W;
        std::optional<KeyboardBinding> backward = controller::Key::S;
        std::optional<KeyboardBinding> turn_left = controller::Key::A;
        std::optional<KeyboardBinding> turn_right = controller::Key::D;
        std::optional<KeyboardBinding> fire = controller::Key::Space;
    };

    // the six analog channels a pad exposes through the window's callbacks
    enum class GamepadAxis {
        LeftStickX,
        LeftStickY,
        RightStickX,
        RightStickY,
        LeftTrigger,
        RightTrigger
    };

    inline constexpr int NB_GAMEPAD_AXES = 6;

    // One analog slot. `sign` keeps the direction captured for the one-way
    // actions (accelerate / reverse read max(0, sign * value)); the two-way
    // actions (steer, aim) ignore it and stay at +1.
    struct GamepadAxisBinding {
        GamepadAxis axis;
        float sign = 1.f;

        bool operator==(const GamepadAxisBinding &) const = default;
    };

    struct GamepadBindings {
        std::optional<controller::GamepadButton> fire = controller::GamepadButton::RB;
        std::optional<GamepadAxisBinding> steer = GamepadAxisBinding{GamepadAxis::LeftStickX};
        std::optional<GamepadAxisBinding> aim_x = GamepadAxisBinding{GamepadAxis::RightStickX};
        std::optional<GamepadAxisBinding> aim_y = GamepadAxisBinding{GamepadAxis::RightStickY};
        std::optional<GamepadAxisBinding> accelerate =
            GamepadAxisBinding{GamepadAxis::RightTrigger};
        std::optional<GamepadAxisBinding> reverse = GamepadAxisBinding{GamepadAxis::LeftTrigger};
        // preferred pad across reconnections; empty = first connected
        std::string device_guid;
        // display only, shown while the preferred pad is unplugged
        std::string device_name;
    };

    struct ControlBindings {
        KeyboardBindings keyboard;
        GamepadBindings gamepad;
    };

    // canonical names for persistence ("" / nullopt on unknown)
    const char *to_string(GamepadAxis axis);
    std::optional<GamepadAxis> gamepad_axis_from_string(std::string_view name);

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CONTROLLER_BINDINGS_H
