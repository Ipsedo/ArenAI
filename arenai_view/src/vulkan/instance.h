//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_INSTANCE_H
#define ARENAI_VK_INSTANCE_H

#include <vector>

#include "./vk.h"

namespace arenai::view {

    class VulkanInstance {
    public:
        // extra_extensions: e.g. the surface extensions reported by GLFW for
        // the windowed backend; the headless backend passes none.
        // Setting the ARENAI_VK_VALIDATION environment variable enables the
        // Khronos validation layer when it is installed.
        explicit VulkanInstance(const std::vector<const char *> &extra_extensions = {});

        VulkanInstance(const VulkanInstance &) = delete;
        VulkanInstance &operator=(const VulkanInstance &) = delete;

        VkInstance handle() const;
        // the Vulkan version negotiated with the loader (>= 1.2)
        uint32_t api_version() const;

        ~VulkanInstance();

    private:
        VkInstance instance_;
        uint32_t api_version_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_INSTANCE_H
