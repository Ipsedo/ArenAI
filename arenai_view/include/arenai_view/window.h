//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_WINDOW_H
#define ARENAI_WINDOW_H

#include <functional>
#include <memory>
#include <utility>

#include <arenai_controller/callback.h>

using namespace arenai;

namespace arenai::view {

    class AbstractWindow {
    public:
        virtual ~AbstractWindow() = default;

        virtual bool should_close() = 0;
        virtual void poll_events() = 0;

        virtual void set_keyboard_callback(
            const std::shared_ptr<controller::AbstractKeyboardCallback> &callback) = 0;
        virtual void set_gamepad_callback(
            const std::shared_ptr<controller::AbstractGamepadCallback> &callback) = 0;

        virtual void set_resize_callback(std::function<void(int width, int height)> callback) = 0;

        virtual void set_cursor_mode(controller::CursorMode mode) = 0;
        virtual void set_cursor_position(double x, double y) = 0;

        // switches between borderless fullscreen (primary monitor, current
        // video mode) and windowed mode; the new size reaches the application
        // through the resize callback, like any user resize
        virtual void set_fullscreen(bool fullscreen) = 0;
    };

}// namespace arenai::view

#endif// ARENAI_WINDOW_H
