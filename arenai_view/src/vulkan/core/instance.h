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
        explicit VulkanInstance(const std::vector<const char *> &extra_extensions = {});

        VulkanInstance(const VulkanInstance &) = delete;
        VulkanInstance &operator=(const VulkanInstance &) = delete;

        VkInstance handle() const;

        uint32_t api_version() const;

        ~VulkanInstance();

    private:
        VkInstance instance_;
        uint32_t api_version_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_INSTANCE_H
