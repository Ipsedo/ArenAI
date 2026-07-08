//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_WINDOW_H
#define ARENAI_WINDOW_H

#include <memory>
#include <utility>

#include "./input.h"

namespace arenai::view {

    // Observer the application implements to receive raw window/input events.
    // The window (owned by the view backend) drives these callbacks.
    class AbstractWindowCallback {
    public:
        virtual ~AbstractWindowCallback() = default;

        virtual void on_key(Key key, InputAction action) = 0;
        virtual void on_mouse_move(double x, double y) = 0;
        virtual void on_mouse_button(MouseButton button, InputAction action) = 0;
    };

    // On-screen window abstraction: main-loop driving, cursor control and sizing.
    // Presentation (buffer swap) is handled by the PlayerRenderer, not here.

    struct window_sizes {
        int width, height;
    };

    class AbstractWindow {
    public:
        virtual ~AbstractWindow() = default;

        virtual bool should_close() = 0;
        virtual void poll_events() = 0;

        virtual void set_callback(const std::shared_ptr<AbstractWindowCallback> &callback) = 0;

        virtual window_sizes size() const = 0;
        virtual void set_cursor_mode(CursorMode mode) = 0;
        virtual void set_cursor_position(double x, double y) = 0;
    };

}// namespace arenai::view

#endif// ARENAI_WINDOW_H
