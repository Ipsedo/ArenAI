//
// Created by samuel on 08/07/2026.
//

#include "./glfw_window.h"

#include <iostream>
#include <stdexcept>

namespace arenai::view {

    namespace {
        Key to_key(const int glfw_key) {
            switch (glfw_key) {
                case GLFW_KEY_W: return Key::W;
                case GLFW_KEY_A: return Key::A;
                case GLFW_KEY_S: return Key::S;
                case GLFW_KEY_D: return Key::D;
                case GLFW_KEY_SPACE: return Key::Space;
                case GLFW_KEY_ESCAPE: return Key::Escape;
                default: return Key::Unknown;
            }
        }

        MouseButton to_mouse_button(const int glfw_button) {
            switch (glfw_button) {
                case GLFW_MOUSE_BUTTON_RIGHT: return MouseButton::Right;
                case GLFW_MOUSE_BUTTON_MIDDLE: return MouseButton::Middle;
                default: return MouseButton::Left;
            }
        }

        InputAction to_action(const int glfw_action) {
            switch (glfw_action) {
                case GLFW_RELEASE: return InputAction::Release;
                case GLFW_REPEAT: return InputAction::Repeat;
                default: return InputAction::Press;
            }
        }
    }// namespace

    GlfwWindow::GlfwWindow(const int width, const int height, const std::string &title) {
        glfwSetErrorCallback([](const int error, const char *description) -> void {
            std::cerr << "GLFW error " << error << ": " << description << std::endl;
        });

        if (!glfwInit()) throw std::runtime_error("glfwInit() failed");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

        window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (!window_) {
            glfwTerminate();
            throw std::runtime_error("glfwCreateWindow() failed");
        }

        glfwMakeContextCurrent(window_);

        glfwSetWindowUserPointer(window_, this);

        glfwSetKeyCallback(
            window_, [](GLFWwindow *w, const int key, int, const int action, int) -> void {
                static_cast<GlfwWindow *>(glfwGetWindowUserPointer(w))->on_key(key, action);
            });

        glfwSetCursorPosCallback(
            window_, [](GLFWwindow *w, const double x, const double y) -> void {
                static_cast<GlfwWindow *>(glfwGetWindowUserPointer(w))->on_cursor(x, y);
            });

        glfwSetMouseButtonCallback(
            window_, [](GLFWwindow *w, const int button, const int action, int) -> void {
                static_cast<GlfwWindow *>(glfwGetWindowUserPointer(w))
                    ->on_mouse_button(button, action);
            });

        glfwSetFramebufferSizeCallback(
            window_, [](GLFWwindow *w, const int new_width, const int new_height) -> void {
                static_cast<GlfwWindow *>(glfwGetWindowUserPointer(w))
                    ->on_resize(new_width, new_height);
            });
    }

    GlfwWindow::~GlfwWindow() {
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

    void GlfwWindow::on_key(const int key, const int action) const {
        if (callback_) callback_->on_key(to_key(key), to_action(action));
    }

    void GlfwWindow::on_cursor(const double x, const double y) const {
        if (callback_) callback_->on_mouse_move(x, y);
    }

    void GlfwWindow::on_mouse_button(const int button, const int action) const {
        if (callback_) callback_->on_mouse_button(to_mouse_button(button), to_action(action));
    }

    void GlfwWindow::on_resize(const int width, const int height) const {
        if (resize_callback_) resize_callback_(width, height);
    }

    bool GlfwWindow::should_close() { return glfwWindowShouldClose(window_); }

    void GlfwWindow::poll_events() { glfwPollEvents(); }

    void GlfwWindow::set_callback(const std::shared_ptr<AbstractWindowCallback> &callback) {
        callback_ = callback;
    }

    void GlfwWindow::set_resize_callback(std::function<void(int width, int height)> callback) {
        resize_callback_ = std::move(callback);
    }

    void GlfwWindow::set_cursor_mode(const CursorMode mode) {
        glfwSetInputMode(
            window_, GLFW_CURSOR,
            mode == CursorMode::Disabled ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }

    void GlfwWindow::set_cursor_position(const double x, const double y) {
        glfwSetCursorPos(window_, x, y);
    }

    EGLDisplay GlfwWindow::egl_display() const { return glfwGetEGLDisplay(); }

    EGLSurface GlfwWindow::egl_surface() const { return glfwGetEGLSurface(window_); }

    EGLContext GlfwWindow::egl_context() const { return glfwGetEGLContext(window_); }

}// namespace arenai::view
