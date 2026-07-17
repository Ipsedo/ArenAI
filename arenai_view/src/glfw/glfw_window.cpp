//
// Created by samuel on 08/07/2026.
//

#include "./glfw_window.h"

#include <iostream>
#include <stdexcept>

namespace arenai::view {

    namespace {
        controller::Key to_key(const int glfw_key) {
            switch (glfw_key) {
                case GLFW_KEY_W: return controller::Key::W;
                case GLFW_KEY_A: return controller::Key::A;
                case GLFW_KEY_S: return controller::Key::S;
                case GLFW_KEY_D: return controller::Key::D;
                case GLFW_KEY_SPACE: return controller::Key::Space;
                case GLFW_KEY_ESCAPE: return controller::Key::Escape;
                default: return controller::Key::Unknown;
            }
        }

        controller::MouseButton to_mouse_button(const int glfw_button) {
            switch (glfw_button) {
                case GLFW_MOUSE_BUTTON_RIGHT: return controller::MouseButton::Right;
                case GLFW_MOUSE_BUTTON_MIDDLE: return controller::MouseButton::Middle;
                default: return controller::MouseButton::Left;
            }
        }

        controller::InputAction to_action(const int glfw_action) {
            switch (glfw_action) {
                case GLFW_RELEASE: return controller::InputAction::Release;
                case GLFW_REPEAT: return controller::InputAction::Repeat;
                default: return controller::InputAction::Press;
            }
        }

        constexpr std::pair<int, controller::GamepadButton> GAMEPAD_BUTTONS[] = {
            {GLFW_GAMEPAD_BUTTON_A, controller::GamepadButton::A},
            {GLFW_GAMEPAD_BUTTON_B, controller::GamepadButton::B},
            {GLFW_GAMEPAD_BUTTON_X, controller::GamepadButton::X},
            {GLFW_GAMEPAD_BUTTON_Y, controller::GamepadButton::Y},
            {GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER, controller::GamepadButton::RB},
            {GLFW_GAMEPAD_BUTTON_LEFT_BUMPER, controller::GamepadButton::LB},
            {GLFW_GAMEPAD_BUTTON_START, controller::GamepadButton::Start},
        };

        // GLFW only exposes the gamepad API for joysticks whose GUID it finds in
        // its bundled SDL mapping database, which is too old to know some common
        // controllers (e.g. the Xbox One S over Bluetooth via the xpadneo driver,
        // GUID 050000005e0400008e02000030110000). Supplement it here so those pads
        // are recognized; a GUID already known to GLFW is simply overridden.
        constexpr auto EXTRA_GAMEPAD_MAPPINGS =
            "050000005e0400008e02000030110000,Xbox Wireless Controller,"
            "a:b0,b:b1,x:b2,y:b3,back:b6,guide:b8,start:b7,"
            "leftstick:b9,rightstick:b10,leftshoulder:b4,rightshoulder:b5,"
            "dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,"
            "leftx:a0,lefty:a1,rightx:a3,righty:a4,lefttrigger:a2,righttrigger:a5,platform:Linux,"
            "\n";
    }// namespace

    GlfwWindow::GlfwWindow(const int width, const int height, const std::string &title) {
        glfwSetErrorCallback([](const int error, const char *description) -> void {
            std::cerr << "GLFW error " << error << ": " << description << std::endl;
        });

        if (!glfwInit()) throw std::runtime_error("glfwInit() failed");

        glfwUpdateGamepadMappings(EXTRA_GAMEPAD_MAPPINGS);

        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        // keep EGL context creation so glfwGetEGL{Display,Surface,Context} stay
        // usable to build the NativeEglContext bridging the window to the renderer
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

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

        glfwSetScrollCallback(
            window_, [](GLFWwindow *w, const double x_offset, const double y_offset) -> void {
                static_cast<GlfwWindow *>(glfwGetWindowUserPointer(w))
                    ->on_scroll(x_offset, y_offset);
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
        if (keyboard_callback_) keyboard_callback_->on_key(to_key(key), to_action(action));
    }

    void GlfwWindow::on_cursor(const double x, const double y) const {
        if (keyboard_callback_) keyboard_callback_->on_mouse_move(x, y);
    }

    void GlfwWindow::on_mouse_button(const int button, const int action) const {
        if (keyboard_callback_)
            keyboard_callback_->on_mouse_button(to_mouse_button(button), to_action(action));
    }

    void GlfwWindow::on_scroll(const double x_offset, const double y_offset) const {
        if (keyboard_callback_) keyboard_callback_->on_scroll(x_offset, y_offset);
    }

    void GlfwWindow::on_resize(const int width, const int height) const {
        if (resize_callback_) resize_callback_(width, height);
    }

    bool GlfwWindow::should_close() { return glfwWindowShouldClose(window_); }

    void GlfwWindow::poll_events() {
        glfwPollEvents();

        // gamepads have no GLFW event callbacks, their state has to be polled
        poll_gamepad();
    }

    void GlfwWindow::poll_gamepad() {
        if (!gamepad_callback_) return;

        GLFWgamepadstate state;
        bool found = false;
        for (int jid = GLFW_JOYSTICK_1; jid <= GLFW_JOYSTICK_LAST && !found; jid++)
            found = glfwJoystickIsGamepad(jid) && glfwGetGamepadState(jid, &state);
        if (!found) {
            // a joystick GLFW has no gamepad mapping for would be silently
            // ignored above: report it once to ease diagnosis
            for (int jid = GLFW_JOYSTICK_1; jid <= GLFW_JOYSTICK_LAST && !unmapped_joystick_warned_;
                 jid++)
                if (glfwJoystickPresent(jid)) {
                    std::cerr << "Joystick \"" << glfwGetJoystickName(jid)
                              << "\" has no gamepad mapping, ignoring it" << std::endl;
                    unmapped_joystick_warned_ = true;
                }
            return;
        }

        for (const auto &[glfw_button, button]: GAMEPAD_BUTTONS) {
            const auto curr = state.buttons[glfw_button];

            if (const auto prev = gamepad_button_states_[glfw_button];
                curr == GLFW_PRESS && prev == GLFW_RELEASE)
                gamepad_callback_->on_gamepad_button(button, controller::InputAction::Press);
            else if (curr == GLFW_RELEASE && prev == GLFW_PRESS)
                gamepad_callback_->on_gamepad_button(button, controller::InputAction::Release);

            gamepad_button_states_[glfw_button] = curr;
        }

        // GLFW triggers range in [-1, 1], normalized here to [0, 1]
        gamepad_callback_->on_trigger(
            (state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] + 1.) / 2.,
            controller::GamepadTrigger::Left);
        gamepad_callback_->on_trigger(
            (state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.) / 2.,
            controller::GamepadTrigger::Right);

        // right stick last: it is dispatched exactly once per frame, so handlers
        // may use it as the per-frame tick to apply the accumulated input state
        gamepad_callback_->on_joystick(
            state.axes[GLFW_GAMEPAD_AXIS_LEFT_X], state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y],
            controller::GamepadJoystick::Left);
        gamepad_callback_->on_joystick(
            state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y],
            controller::GamepadJoystick::Right);
    }

    void GlfwWindow::set_keyboard_callback(
        const std::shared_ptr<controller::AbstractKeyboardCallback> &callback) {
        keyboard_callback_ = callback;
    }

    void GlfwWindow::set_gamepad_callback(
        const std::shared_ptr<controller::AbstractGamepadCallback> &callback) {
        gamepad_callback_ = callback;
    }

    void GlfwWindow::set_resize_callback(std::function<void(int width, int height)> callback) {
        resize_callback_ = std::move(callback);
    }

    void GlfwWindow::set_cursor_mode(const controller::CursorMode mode) {
        glfwSetInputMode(
            window_, GLFW_CURSOR,
            mode == controller::CursorMode::Disabled ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }

    void GlfwWindow::set_cursor_position(const double x, const double y) {
        glfwSetCursorPos(window_, x, y);
    }

    EGLDisplay GlfwWindow::egl_display() const { return glfwGetEGLDisplay(); }

    EGLSurface GlfwWindow::egl_surface() const { return glfwGetEGLSurface(window_); }

    EGLContext GlfwWindow::egl_context() const { return glfwGetEGLContext(window_); }

}// namespace arenai::view
