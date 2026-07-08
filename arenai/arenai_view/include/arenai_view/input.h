//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_INPUT_H
#define ARENAI_INPUT_H

namespace arenai::view {

    // Platform-agnostic raw input vocabulary. The windowing backend translates
    // its native codes (GLFW, ...) into these; consumers never see a backend enum.
    enum class Key { Unknown, W, A, S, D, Space, Escape };

    enum class MouseButton { Left, Right, Middle };

    enum class InputAction { Press, Release, Repeat };

    enum class CursorMode { Normal, Disabled };

}// namespace arenai::view

#endif// ARENAI_INPUT_H
