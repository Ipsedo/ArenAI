//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_GLFW_VULKAN_WINDOW_H
#define ARENAI_GLFW_VULKAN_WINDOW_H

#include <array>
#include <functional>
#include <memory>
#include <string>

// GLFW pulls no GL header (GLFW_INCLUDE_NONE); vulkan.h (via vk.h) MUST come
// first so glfw3.h declares its Vulkan helpers (glfwCreateWindowSurface, ...)
// clang-format off
#define GLFW_INCLUDE_NONE
#include "../vulkan/vk.h"
#include <GLFW/glfw3.h>
// clang-format on

#include "../vulkan/vulkan_window.h"

namespace arenai::view {

    class GlfwVulkanWindow final : public AbstractVulkanWindow {
    public:
        GlfwVulkanWindow(int width, int height, const std::string &title);
        ~GlfwVulkanWindow() override;

        bool should_close() override;
        void poll_events() override;

        void set_keyboard_callback(
            const std::shared_ptr<controller::AbstractKeyboardCallback> &callback) override;
        void set_gamepad_callback(
            const std::shared_ptr<controller::AbstractGamepadCallback> &callback) override;

        void set_resize_callback(std::function<void(int width, int height)> callback) override;

        void set_cursor_mode(controller::CursorMode mode) override;
        void set_cursor_position(double x, double y) override;

        void set_fullscreen(bool fullscreen) override;

        std::vector<const char *> required_instance_extensions() const override;
        VkSurfaceKHR create_surface(VkInstance instance) const override;

        std::tuple<int, int> framebuffer_size() const override;

    private:
        GLFWwindow *window_;
        std::shared_ptr<controller::AbstractKeyboardCallback> keyboard_callback_;
        std::shared_ptr<controller::AbstractGamepadCallback> gamepad_callback_;

        std::function<void(int width, int height)> resize_callback_;

        std::array<unsigned char, GLFW_GAMEPAD_BUTTON_LAST + 1> gamepad_button_states_{};
        bool unmapped_joystick_warned_ = false;

        // windowed geometry, saved on entering fullscreen and restored on exit
        int windowed_x_ = 0, windowed_y_ = 0;
        int windowed_width_ = 0, windowed_height_ = 0;

        void on_key(int key, int action) const;
        void on_cursor(double x, double y) const;
        void on_mouse_button(int button, int action) const;
        void on_scroll(double x_offset, double y_offset) const;
        void on_resize(int width, int height) const;

        void poll_gamepad();
    };

}// namespace arenai::view

#endif// ARENAI_GLFW_VULKAN_WINDOW_H
