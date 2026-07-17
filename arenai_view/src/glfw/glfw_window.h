//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_GLFW_WINDOW_H
#define ARENAI_GLFW_WINDOW_H

#include <array>
#include <functional>
#include <memory>
#include <string>

#define GLFW_EXPOSE_NATIVE_EGL
// GLFW must not pull its own GL headers: the desktop GL API is included
// centrally through src/opengl/gl.h, so let that be the single source
#define GLFW_INCLUDE_NONE
#include <EGL/egl.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "../opengl/gl_window.h"

namespace arenai::view {

    class GlfwWindow final : public AbstractGlWindow {
    public:
        GlfwWindow(int width, int height, const std::string &title);
        ~GlfwWindow() override;

        bool should_close() override;
        void poll_events() override;

        void set_keyboard_callback(
            const std::shared_ptr<controller::AbstractKeyboardCallback> &callback) override;
        void set_gamepad_callback(
            const std::shared_ptr<controller::AbstractGamepadCallback> &callback) override;

        void set_resize_callback(std::function<void(int width, int height)> callback) override;

        void set_cursor_mode(controller::CursorMode mode) override;
        void set_cursor_position(double x, double y) override;

        EGLDisplay egl_display() const override;
        EGLSurface egl_surface() const override;
        EGLContext egl_context() const override;

    private:
        GLFWwindow *window_;
        std::shared_ptr<controller::AbstractKeyboardCallback> keyboard_callback_;
        std::shared_ptr<controller::AbstractGamepadCallback> gamepad_callback_;

        std::function<void(int width, int height)> resize_callback_;

        std::array<unsigned char, GLFW_GAMEPAD_BUTTON_LAST + 1> gamepad_button_states_{};
        bool unmapped_joystick_warned_ = false;

        void on_key(int key, int action) const;
        void on_cursor(double x, double y) const;
        void on_mouse_button(int button, int action) const;
        void on_scroll(double x_offset, double y_offset) const;
        void on_resize(int width, int height) const;

        void poll_gamepad();
    };

}// namespace arenai::view

#endif// ARENAI_GLFW_WINDOW_H
