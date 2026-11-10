#ifndef ARENAI_INPUT_NAMES_H
#define ARENAI_INPUT_NAMES_H

#include <optional>
#include <string_view>

#include "./callback.h"

namespace arenai::controller {

    const char *to_string(Key key);
    Key key_from_string(std::string_view name);

    const char *to_string(GamepadButton button);
    std::optional<GamepadButton> gamepad_button_from_string(std::string_view name);

}// namespace arenai::controller

#endif// ARENAI_INPUT_NAMES_H
