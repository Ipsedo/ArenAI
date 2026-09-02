//
// Created by samuel on 27/08/2026.
//

#include "./bindings.h"

#include <utility>

namespace arenai::desktop {

    namespace {

        constexpr std::pair<GamepadAxis, const char *> AXIS_NAMES[] = {
            {GamepadAxis::LeftStickX, "LeftStickX"},   {GamepadAxis::LeftStickY, "LeftStickY"},
            {GamepadAxis::RightStickX, "RightStickX"}, {GamepadAxis::RightStickY, "RightStickY"},
            {GamepadAxis::LeftTrigger, "LeftTrigger"}, {GamepadAxis::RightTrigger, "RightTrigger"},
        };

    }// namespace

    const char *to_string(const GamepadAxis axis) {
        for (const auto &[value, name]: AXIS_NAMES)
            if (value == axis) return name;
        return "";
    }

    std::optional<GamepadAxis> gamepad_axis_from_string(const std::string_view name) {
        for (const auto &[value, axis_name]: AXIS_NAMES)
            if (name == axis_name) return value;
        return std::nullopt;
    }

}// namespace arenai::desktop
