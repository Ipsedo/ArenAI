//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VULKAN_WINDOW_H
#define ARENAI_VULKAN_WINDOW_H

#include <tuple>
#include <vector>

#include <arenai_view/window.h>

#include "./core/vk.h"

namespace arenai::view {

    class AbstractVulkanWindow : public AbstractWindow {
    public:
        virtual std::vector<const char *> required_instance_extensions() const = 0;
        virtual VkSurfaceKHR create_surface(const VkInstance &instance) const = 0;

        virtual std::tuple<int, int> framebuffer_size() const = 0;
    };

}// namespace arenai::view

#endif// ARENAI_VULKAN_WINDOW_H
