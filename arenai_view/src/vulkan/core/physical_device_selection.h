//
// Created by samuel on 20/07/2026.
//

#ifndef ARENAI_VK_PHYSICAL_DEVICE_SELECTION_H
#define ARENAI_VK_PHYSICAL_DEVICE_SELECTION_H

#include <cstdint>

#include "./vk.h"

namespace arenai::view {

    struct DeviceCriteria {
        // vision (headless) backend prefers the integrated GPU, leaving the
        // discrete one to the window; both fall back to whatever exists
        bool prefer_integrated = true;
        // when set, the device must be able to present to this surface and
        // VK_KHR_swapchain is enabled
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        // environment variable overriding the selection (device index or a
        // deviceName substring), e.g. "ARENAI_VK_DEVICE"
        const char *device_env_var = nullptr;
    };

    struct PhysicalDeviceChoice {
        VkPhysicalDevice device;
        VkPhysicalDeviceProperties properties;
        uint32_t queue_family;
    };

    // Picks the physical device matching the criteria: graphics queue
    // (presentable to the surface when one is set), Vulkan 1.2+ with dynamic
    // rendering, ranked by device type according to prefer_integrated. The
    // env var, when set, overrides the ranking. Throws when nothing fits.
    PhysicalDeviceChoice pick_physical_device(VkInstance instance, const DeviceCriteria &criteria);

}// namespace arenai::view

#endif// ARENAI_VK_PHYSICAL_DEVICE_SELECTION_H
