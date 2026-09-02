//
// Created by samuel on 20/07/2026.
//

#include "./physical_device_selection.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "./errors.h"

namespace arenai::view {

    namespace {

        std::optional<uint32_t>
        find_graphics_family(const VkPhysicalDevice &device, const VkSurfaceKHR &surface) {
            uint32_t count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

            for (uint32_t i = 0; i < count; i++) {
                if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
                if (surface != VK_NULL_HANDLE) {
                    VkBool32 can_present = VK_FALSE;
                    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &can_present);
                    if (!can_present) continue;
                }
                return i;
            }
            return std::nullopt;
        }

        bool has_extension(const VkPhysicalDevice &device, const char *name) {
            uint32_t count = 0;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> extensions(count);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
            return std::ranges::any_of(extensions, [name](const VkExtensionProperties &extension) {
                return std::strcmp(extension.extensionName, name) == 0;
            });
        }

        bool supports_dynamic_rendering(const VkPhysicalDevice &device) {
            VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering{};
            dynamic_rendering.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
            VkPhysicalDeviceFeatures2 features{};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &dynamic_rendering;
            vkGetPhysicalDeviceFeatures2(device, &features);
            return dynamic_rendering.dynamicRendering == VK_TRUE;
        }

        // ranks the device type according to the backend preference; lower is better
        int type_rank(const VkPhysicalDeviceType type, const bool prefer_integrated) {
            const auto order =
                prefer_integrated
                    ? std::
                        vector{VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, VK_PHYSICAL_DEVICE_TYPE_CPU}
                    : std::vector{
                        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                        VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, VK_PHYSICAL_DEVICE_TYPE_CPU};
            const auto it = std::ranges::find(order, type);
            return it == order.end() ? static_cast<int>(order.size())
                                     : static_cast<int>(it - order.begin());
        }

        std::vector<PhysicalDeviceChoice>
        eligible_candidates(const VkInstance &instance, const VkSurfaceKHR &surface) {
            uint32_t count = 0;
            vk_check(
                vkEnumeratePhysicalDevices(instance, &count, nullptr),
                "vkEnumeratePhysicalDevices");
            if (count == 0) throw std::runtime_error("no Vulkan physical device found");
            std::vector<VkPhysicalDevice> devices(count);
            vkEnumeratePhysicalDevices(instance, &count, devices.data());

            std::vector<PhysicalDeviceChoice> candidates;
            for (const auto device: devices) {
                VkPhysicalDeviceProperties properties;
                vkGetPhysicalDeviceProperties(device, &properties);

                if (properties.apiVersion < VK_API_VERSION_1_2) continue;
                if (properties.apiVersion < VK_API_VERSION_1_3
                    && !has_extension(device, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME))
                    continue;
                if (!supports_dynamic_rendering(device)) continue;
                if (surface != VK_NULL_HANDLE
                    && !has_extension(device, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
                    continue;

                const auto family = find_graphics_family(device, surface);
                if (!family.has_value()) continue;

                candidates.push_back(
                    {.device = device, .properties = properties, .queue_family = family.value()});
            }
            return candidates;
        }

    }// namespace

    PhysicalDeviceChoice
    pick_physical_device(const VkInstance &instance, const DeviceCriteria &criteria) {
        std::vector<PhysicalDeviceChoice> candidates =
            eligible_candidates(instance, criteria.surface);
        if (candidates.empty())
            throw std::runtime_error(
                "no Vulkan device with graphics + dynamic rendering (1.2+) found");

        // environment override: device index or deviceName substring
        if (criteria.device_env_var != nullptr) {
            if (const char *wanted = std::getenv(criteria.device_env_var)) {
                char *end = nullptr;
                const long index = std::strtol(wanted, &end, 10);
                if (end != wanted && *end == '\0') {
                    if (index >= 0 && index < static_cast<long>(candidates.size()))
                        return candidates[index];
                    throw std::runtime_error(
                        std::string(criteria.device_env_var) + ": device index out of range");
                }
                for (const auto &candidate: candidates)
                    if (std::string(candidate.properties.deviceName).find(wanted)
                        != std::string::npos)
                        return candidate;
                throw std::runtime_error(
                    std::string(criteria.device_env_var) + ": no device matching '" + wanted + "'");
            }
        }

        // user preference (menu): softer than the env var, an unmatched name
        // falls back to the automatic pick
        if (!criteria.preferred_device.empty()) {
            for (const auto &candidate: candidates)
                if (std::string(candidate.properties.deviceName).find(criteria.preferred_device)
                    != std::string::npos)
                    return candidate;
            std::cerr << "preferred GPU '" << criteria.preferred_device
                      << "' not found, falling back to automatic selection" << std::endl;
        }

        std::ranges::stable_sort(
            candidates, [&](const PhysicalDeviceChoice &a, const PhysicalDeviceChoice &b) {
                return type_rank(a.properties.deviceType, criteria.prefer_integrated)
                       < type_rank(b.properties.deviceType, criteria.prefer_integrated);
            });
        return candidates.front();
    }

    std::vector<std::string> list_device_names(const VkInstance &instance) {
        std::vector<std::string> names;
        for (const auto &candidate: eligible_candidates(instance, VK_NULL_HANDLE))
            names.emplace_back(candidate.properties.deviceName);
        return names;
    }

}// namespace arenai::view
