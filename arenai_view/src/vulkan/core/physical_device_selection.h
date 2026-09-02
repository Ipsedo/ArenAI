//
// Created by samuel on 20/07/2026.
//

#ifndef ARENAI_VK_PHYSICAL_DEVICE_SELECTION_H
#define ARENAI_VK_PHYSICAL_DEVICE_SELECTION_H

#include <cstdint>
#include <string>
#include <vector>

#include "./vk.h"

namespace arenai::view {

    struct DeviceCriteria {
        bool prefer_integrated = true;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        const char *device_env_var = nullptr;
        // deviceName substring picked by the user (e.g. from the menu); empty =
        // automatic selection. Unlike the env var, an unmatched name falls back
        // to automatic selection: an unplugged GPU must not prevent startup.
        std::string preferred_device;
    };

    struct PhysicalDeviceChoice {
        VkPhysicalDevice device;
        VkPhysicalDeviceProperties properties;
        uint32_t queue_family;
    };

    PhysicalDeviceChoice
    pick_physical_device(const VkInstance &instance, const DeviceCriteria &criteria);

    // deviceName of every device pick_physical_device would consider for an
    // offscreen use (surface-less filters), in Vulkan enumeration order
    std::vector<std::string> list_device_names(const VkInstance &instance);

}// namespace arenai::view

#endif// ARENAI_VK_PHYSICAL_DEVICE_SELECTION_H
