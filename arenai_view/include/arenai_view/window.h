#ifndef ARENAI_WINDOW_H
#define ARENAI_WINDOW_H

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <arenai_controller/callback.h>

namespace arenai::view {

    struct GamepadInfo {
        int id;
        std::string name;
        std::string guid;
    };

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

        virtual void set_fullscreen(bool fullscreen) = 0;

        virtual std::tuple<int, int> screen_size() const = 0;

        // connected gamepads, in the backend's slot order
        virtual std::vector<GamepadInfo> list_gamepads() const = 0;
        virtual void select_gamepad(int id) = 0;

        virtual std::string key_label(controller::Key key) const = 0;
    };

}// namespace arenai::view

#endif// ARENAI_WINDOW_H
