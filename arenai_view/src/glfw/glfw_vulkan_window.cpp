//
// Created by samuel on 08/07/2026.
//

#include "./glfw_vulkan_window.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>

#include <arenai_controller/input_names.h>

namespace arenai::view {

    namespace {
        // both directions rely on A..Z, 0..9 and F1..F12 being contiguous in
        // GLFW and in controller::Key
        constexpr std::pair<int, controller::Key> SPECIAL_KEYS[] = {
            {GLFW_KEY_UP, controller::Key::Up},
            {GLFW_KEY_DOWN, controller::Key::Down},
            {GLFW_KEY_LEFT, controller::Key::Left},
            {GLFW_KEY_RIGHT, controller::Key::Right},
            {GLFW_KEY_SPACE, controller::Key::Space},
            {GLFW_KEY_ESCAPE, controller::Key::Escape},
            {GLFW_KEY_ENTER, controller::Key::Enter},
            {GLFW_KEY_TAB, controller::Key::Tab},
            {GLFW_KEY_BACKSPACE, controller::Key::Backspace},
            {GLFW_KEY_INSERT, controller::Key::Insert},
            {GLFW_KEY_DELETE, controller::Key::Delete},
            {GLFW_KEY_HOME, controller::Key::Home},
            {GLFW_KEY_END, controller::Key::End},
            {GLFW_KEY_PAGE_UP, controller::Key::PageUp},
            {GLFW_KEY_PAGE_DOWN, controller::Key::PageDown},
            {GLFW_KEY_LEFT_SHIFT, controller::Key::LeftShift},
            {GLFW_KEY_RIGHT_SHIFT, controller::Key::RightShift},
            {GLFW_KEY_LEFT_CONTROL, controller::Key::LeftControl},
            {GLFW_KEY_RIGHT_CONTROL, controller::Key::RightControl},
            {GLFW_KEY_LEFT_ALT, controller::Key::LeftAlt},
            {GLFW_KEY_RIGHT_ALT, controller::Key::RightAlt},
            {GLFW_KEY_COMMA, controller::Key::Comma},
            {GLFW_KEY_PERIOD, controller::Key::Period},
            {GLFW_KEY_SEMICOLON, controller::Key::Semicolon},
            {GLFW_KEY_APOSTROPHE, controller::Key::Apostrophe},
            {GLFW_KEY_SLASH, controller::Key::Slash},
            {GLFW_KEY_BACKSLASH, controller::Key::Backslash},
            {GLFW_KEY_MINUS, controller::Key::Minus},
            {GLFW_KEY_EQUAL, controller::Key::Equal},
            {GLFW_KEY_LEFT_BRACKET, controller::Key::LeftBracket},
            {GLFW_KEY_RIGHT_BRACKET, controller::Key::RightBracket},
            {GLFW_KEY_GRAVE_ACCENT, controller::Key::Grave},
        };

        controller::Key shift_key(const controller::Key base, const int offset) {
            return static_cast<controller::Key>(static_cast<int>(base) + offset);
        }

        controller::Key to_key(const int glfw_key) {
            if (glfw_key >= GLFW_KEY_A && glfw_key <= GLFW_KEY_Z)
                return shift_key(controller::Key::A, glfw_key - GLFW_KEY_A);
            if (glfw_key >= GLFW_KEY_0 && glfw_key <= GLFW_KEY_9)
                return shift_key(controller::Key::Num0, glfw_key - GLFW_KEY_0);
            if (glfw_key >= GLFW_KEY_F1 && glfw_key <= GLFW_KEY_F12)
                return shift_key(controller::Key::F1, glfw_key - GLFW_KEY_F1);
            for (const auto &[glfw, key]: SPECIAL_KEYS)
                if (glfw == glfw_key) return key;
            return controller::Key::Unknown;
        }

        int to_glfw_key(const controller::Key key) {
            const auto index = static_cast<int>(key);
            const auto offset = [index](const controller::Key base) {
                return index - static_cast<int>(base);
            };
            if (key >= controller::Key::A && key <= controller::Key::Z)
                return GLFW_KEY_A + offset(controller::Key::A);
            if (key >= controller::Key::Num0 && key <= controller::Key::Num9)
                return GLFW_KEY_0 + offset(controller::Key::Num0);
            if (key >= controller::Key::F1 && key <= controller::Key::F12)
                return GLFW_KEY_F1 + offset(controller::Key::F1);
            for (const auto &[glfw, special]: SPECIAL_KEYS)
                if (special == key) return glfw;
            return GLFW_KEY_UNKNOWN;
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
            {GLFW_GAMEPAD_BUTTON_DPAD_UP, controller::GamepadButton::DPadUp},
            {GLFW_GAMEPAD_BUTTON_DPAD_DOWN, controller::GamepadButton::DPadDown},
            {GLFW_GAMEPAD_BUTTON_DPAD_LEFT, controller::GamepadButton::DPadLeft},
            {GLFW_GAMEPAD_BUTTON_DPAD_RIGHT, controller::GamepadButton::DPadRight},
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

    GlfwVulkanWindow::GlfwVulkanWindow(
        const int width, const int height, const std::string &title) {
        glfwSetErrorCallback([](const int error, const char *description) -> void {
            std::cerr << "GLFW error " << error << ": " << description << std::endl;
        });

        if (!glfwInit()) throw std::runtime_error("glfwInit() failed");

        glfwUpdateGamepadMappings(EXTRA_GAMEPAD_MAPPINGS);

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (!window_) {
            glfwTerminate();
            throw std::runtime_error("glfwCreateWindow() failed");
        }

        glfwSetWindowUserPointer(window_, this);

        glfwSetKeyCallback(
            window_, [](GLFWwindow *w, const int key, int, const int action, int) -> void {
                static_cast<GlfwVulkanWindow *>(glfwGetWindowUserPointer(w))->on_key(key, action);
            });

        glfwSetCursorPosCallback(
            window_, [](GLFWwindow *w, const double x, const double y) -> void {
                static_cast<GlfwVulkanWindow *>(glfwGetWindowUserPointer(w))->on_cursor(x, y);
            });

        glfwSetMouseButtonCallback(
            window_, [](GLFWwindow *w, const int button, const int action, int) -> void {
                static_cast<GlfwVulkanWindow *>(glfwGetWindowUserPointer(w))
                    ->on_mouse_button(button, action);
            });

        glfwSetScrollCallback(
            window_, [](GLFWwindow *w, const double x_offset, const double y_offset) -> void {
                static_cast<GlfwVulkanWindow *>(glfwGetWindowUserPointer(w))
                    ->on_scroll(x_offset, y_offset);
            });

        glfwSetFramebufferSizeCallback(
            window_, [](GLFWwindow *w, const int new_width, const int new_height) -> void {
                static_cast<GlfwVulkanWindow *>(glfwGetWindowUserPointer(w))
                    ->on_resize(new_width, new_height);
            });
    }

    GlfwVulkanWindow::~GlfwVulkanWindow() {
        if (window_) glfwDestroyWindow(window_);
        glfwTerminate();
    }

    void GlfwVulkanWindow::on_key(const int key, const int action) const {
        if (keyboard_callback_) keyboard_callback_->on_key(to_key(key), to_action(action));
    }

    void GlfwVulkanWindow::on_cursor(const double x, const double y) const {
        if (keyboard_callback_) keyboard_callback_->on_mouse_move(x, y);
    }

    void GlfwVulkanWindow::on_mouse_button(const int button, const int action) const {
        if (keyboard_callback_)
            keyboard_callback_->on_mouse_button(to_mouse_button(button), to_action(action));
    }

    void GlfwVulkanWindow::on_scroll(const double x_offset, const double y_offset) const {
        if (keyboard_callback_) keyboard_callback_->on_scroll(x_offset, y_offset);
    }

    void GlfwVulkanWindow::on_resize(const int width, const int height) const {
        if (resize_callback_) resize_callback_(width, height);
    }

    bool GlfwVulkanWindow::should_close() { return glfwWindowShouldClose(window_); }

    void GlfwVulkanWindow::poll_events() {
        glfwPollEvents();

        // gamepads have no GLFW event callbacks, their state has to be polled
        poll_gamepad();
    }

    void GlfwVulkanWindow::poll_gamepad() {
        if (!gamepad_callback_) return;

        GLFWgamepadstate state;
        // the selected pad first; unplugged or never selected falls back to
        // the first connected one
        bool found = selected_gamepad_ >= 0 && glfwJoystickIsGamepad(selected_gamepad_)
                     && glfwGetGamepadState(selected_gamepad_, &state);
        for (int jid = GLFW_JOYSTICK_1; jid <= GLFW_JOYSTICK_LAST && !found; jid++)
            found = glfwJoystickIsGamepad(jid) && glfwGetGamepadState(jid, &state);
        if (!found) {
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

        gamepad_callback_->on_trigger(
            (state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER] + 1.) / 2.,
            controller::GamepadTrigger::Left);
        gamepad_callback_->on_trigger(
            (state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.) / 2.,
            controller::GamepadTrigger::Right);

        gamepad_callback_->on_joystick(
            state.axes[GLFW_GAMEPAD_AXIS_LEFT_X], state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y],
            controller::GamepadJoystick::Left);
        gamepad_callback_->on_joystick(
            state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y],
            controller::GamepadJoystick::Right);
    }

    void GlfwVulkanWindow::set_keyboard_callback(
        const std::shared_ptr<controller::AbstractKeyboardCallback> &callback) {
        keyboard_callback_ = callback;
    }

    void GlfwVulkanWindow::set_gamepad_callback(
        const std::shared_ptr<controller::AbstractGamepadCallback> &callback) {
        gamepad_callback_ = callback;
    }

    void
    GlfwVulkanWindow::set_resize_callback(std::function<void(int width, int height)> callback) {
        resize_callback_ = std::move(callback);
    }

    void GlfwVulkanWindow::set_cursor_mode(const controller::CursorMode mode) {
        glfwSetInputMode(
            window_, GLFW_CURSOR,
            mode == controller::CursorMode::Disabled ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }

    void GlfwVulkanWindow::set_cursor_position(const double x, const double y) {
        glfwSetCursorPos(window_, x, y);
    }

    void GlfwVulkanWindow::set_fullscreen(const bool fullscreen) {
        if (const bool is_fullscreen = glfwGetWindowMonitor(window_) != nullptr;
            fullscreen == is_fullscreen)
            return;

        if (fullscreen) {
            glfwGetWindowPos(window_, &windowed_x_, &windowed_y_);
            glfwGetWindowSize(window_, &windowed_width_, &windowed_height_);

            GLFWmonitor *monitor = glfwGetPrimaryMonitor();
            const GLFWvidmode *mode = glfwGetVideoMode(monitor);
            glfwSetWindowMonitor(
                window_, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
        } else {
            glfwSetWindowMonitor(
                window_, nullptr, windowed_x_, windowed_y_, windowed_width_, windowed_height_, 0);
        }
    }

    std::tuple<int, int> GlfwVulkanWindow::screen_size() const {
        GLFWmonitor *monitor = glfwGetWindowMonitor(window_);

        if (!monitor) {
            int x, y, width, height;
            glfwGetWindowPos(window_, &x, &y);
            glfwGetWindowSize(window_, &width, &height);

            int count = 0;
            GLFWmonitor **monitors = glfwGetMonitors(&count);
            int best_overlap = 0;
            for (int i = 0; i < count; i++) {
                const GLFWvidmode *mode = glfwGetVideoMode(monitors[i]);
                if (!mode) continue;
                int monitor_x, monitor_y;
                glfwGetMonitorPos(monitors[i], &monitor_x, &monitor_y);

                const int overlap =
                    std::max(
                        0, std::min(x + width, monitor_x + mode->width) - std::max(x, monitor_x))
                    * std::max(
                        0, std::min(y + height, monitor_y + mode->height) - std::max(y, monitor_y));
                if (overlap > best_overlap) {
                    best_overlap = overlap;
                    monitor = monitors[i];
                }
            }

            if (!monitor) monitor = glfwGetPrimaryMonitor();
        }

        if (monitor)
            if (const GLFWvidmode *mode = glfwGetVideoMode(monitor))
                return {mode->width, mode->height};

        return framebuffer_size();
    }

    std::vector<GamepadInfo> GlfwVulkanWindow::list_gamepads() const {
        std::vector<GamepadInfo> gamepads;
        for (int jid = GLFW_JOYSTICK_1; jid <= GLFW_JOYSTICK_LAST; jid++) {
            if (!glfwJoystickIsGamepad(jid)) continue;
            const char *name = glfwGetGamepadName(jid);
            const char *guid = glfwGetJoystickGUID(jid);
            gamepads.push_back(
                {.id = jid,
                 .name = name != nullptr ? name : "Gamepad",
                 .guid = guid != nullptr ? guid : ""});
        }
        return gamepads;
    }

    void GlfwVulkanWindow::select_gamepad(const int id) { selected_gamepad_ = id; }

    std::string GlfwVulkanWindow::key_label(const controller::Key key) const {
        // layout-aware name for printable keys (Key::Q reads "A" on AZERTY);
        // GLFW returns nullptr for the non-printable ones
        if (const int glfw_key = to_glfw_key(key); glfw_key != GLFW_KEY_UNKNOWN)
            if (const char *name = glfwGetKeyName(glfw_key, 0); name != nullptr && *name != '\0') {
                std::string label(name);
                std::ranges::transform(label, label.begin(), [](const unsigned char c) {
                    return static_cast<char>(std::toupper(c));
                });
                return label;
            }
        return controller::to_string(key);
    }

    std::vector<const char *> GlfwVulkanWindow::required_instance_extensions() const {
        uint32_t count = 0;
        const char **extensions = glfwGetRequiredInstanceExtensions(&count);
        if (!extensions) throw std::runtime_error("GLFW reports no Vulkan support on this system");
        return {extensions, extensions + count};
    }

    VkSurfaceKHR GlfwVulkanWindow::create_surface(const VkInstance &instance) const {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (glfwCreateWindowSurface(instance, window_, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("glfwCreateWindowSurface() failed");
        return surface;
    }

    std::tuple<int, int> GlfwVulkanWindow::framebuffer_size() const {
        int width, height;
        glfwGetFramebufferSize(window_, &width, &height);
        return {width, height};
    }

}// namespace arenai::view
