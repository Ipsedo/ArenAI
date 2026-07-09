//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_WINDOW_H
#define ARENAI_WINDOW_H

#include <functional>
#include <memory>
#include <utility>

namespace arenai::view {

    enum class Key { Unknown, W, A, S, D, Space, Escape };
    enum class MouseButton { Left, Right, Middle };
    enum class InputAction { Press, Release, Repeat };
    enum class CursorMode { Normal, Disabled };

    class AbstractWindowCallback {
    public:
        virtual ~AbstractWindowCallback() = default;

        virtual void on_key(Key key, InputAction action) = 0;
        virtual void on_mouse_move(double x, double y) = 0;
        virtual void on_mouse_button(MouseButton button, InputAction action) = 0;
    };

    class AbstractWindow {
    public:
        virtual ~AbstractWindow() = default;

        virtual bool should_close() = 0;
        virtual void poll_events() = 0;

        virtual void set_callback(const std::shared_ptr<AbstractWindowCallback> &callback) = 0;
        virtual void set_resize_callback(std::function<void(int width, int height)> callback) = 0;

        virtual void set_cursor_mode(CursorMode mode) = 0;
        virtual void set_cursor_position(double x, double y) = 0;
    };

}// namespace arenai::view

#endif// ARENAI_WINDOW_H
