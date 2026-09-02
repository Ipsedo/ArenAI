//
// Created by samuel on 27/08/2026.
//

#ifndef ARENAI_INPUT_NAMES_H
#define ARENAI_INPUT_NAMES_H

#include <optional>
#include <string_view>

#include "./callback.h"

// Canonical, layout-independent names for the input enums: stable identifiers
// for persistence, and display fallbacks when the windowing backend has no
// layout-aware label for a key.
namespace arenai::controller {

    // "" for Key::Unknown
    const char *to_string(Key key);
    // Key::Unknown when the name is not recognized
    Key key_from_string(std::string_view name);

    const char *to_string(GamepadButton button);
    std::optional<GamepadButton> gamepad_button_from_string(std::string_view name);

}// namespace arenai::controller

#endif// ARENAI_INPUT_NAMES_H
