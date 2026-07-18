//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VULKAN_WINDOW_H
#define ARENAI_VULKAN_WINDOW_H

#include <tuple>
#include <vector>

#include <arenai_view/window.h>

#include "./vk.h"

namespace arenai::view {

    // Window-side contract of the Vulkan backend (the counterpart of the old
    // AbstractGlWindow): the window exposes what instance extensions it needs
    // and turns itself into a VkSurfaceKHR.
    class AbstractVulkanWindow : public AbstractWindow {
    public:
        virtual std::vector<const char *> required_instance_extensions() const = 0;
        virtual VkSurfaceKHR create_surface(VkInstance instance) const = 0;

        virtual std::tuple<int, int> framebuffer_size() const = 0;
    };

}// namespace arenai::view

#endif// ARENAI_VULKAN_WINDOW_H
