//
// Created by samuel on 20/07/2026.
//

#ifndef ARENAI_VK_PHYSICAL_DEVICE_SELECTION_H
#define ARENAI_VK_PHYSICAL_DEVICE_SELECTION_H

#include <cstdint>

#include "./vk.h"

namespace arenai::view {

    struct DeviceCriteria {
        bool prefer_integrated = true;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        const char *device_env_var = nullptr;
    };

    struct PhysicalDeviceChoice {
        VkPhysicalDevice device;
        VkPhysicalDeviceProperties properties;
        uint32_t queue_family;
    };

    PhysicalDeviceChoice
    pick_physical_device(const VkInstance &instance, const DeviceCriteria &criteria);

}// namespace arenai::view

#endif// ARENAI_VK_PHYSICAL_DEVICE_SELECTION_H
