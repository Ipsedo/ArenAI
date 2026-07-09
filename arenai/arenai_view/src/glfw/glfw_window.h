//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_GLFW_WINDOW_H
#define ARENAI_GLFW_WINDOW_H

#include <functional>
#include <memory>
#include <string>

#define GLFW_EXPOSE_NATIVE_EGL
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

        void set_callback(const std::shared_ptr<AbstractWindowCallback> &callback) override;
        void set_resize_callback(std::function<void(int width, int height)> callback) override;

        void set_cursor_mode(CursorMode mode) override;
        void set_cursor_position(double x, double y) override;

        EGLDisplay egl_display() const override;
        EGLSurface egl_surface() const override;
        EGLContext egl_context() const override;

    private:
        GLFWwindow *window_;
        std::shared_ptr<AbstractWindowCallback> callback_;
        std::function<void(int width, int height)> resize_callback_;

        void on_key(int key, int action) const;
        void on_cursor(double x, double y) const;
        void on_mouse_button(int button, int action) const;
        void on_resize(int width, int height) const;
    };

}// namespace arenai::view

#endif// ARENAI_GLFW_WINDOW_H
