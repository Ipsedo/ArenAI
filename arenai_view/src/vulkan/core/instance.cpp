//
// Created by samuel on 17/07/2026.
//

#include "./instance.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "./errors.h"

namespace arenai::view {

    namespace {
        bool validation_layer_available() {
            uint32_t count = 0;
            vkEnumerateInstanceLayerProperties(&count, nullptr);
            std::vector<VkLayerProperties> layers(count);
            vkEnumerateInstanceLayerProperties(&count, layers.data());
            return std::ranges::any_of(layers, [](const VkLayerProperties &layer) {
                return std::string(layer.layerName) == "VK_LAYER_KHRONOS_validation";
            });
        }
    }// namespace

    VulkanInstance::VulkanInstance(const std::vector<const char *> &extra_extensions)
        : instance_(VK_NULL_HANDLE), api_version_(VK_API_VERSION_1_2) {
        // negotiate the highest version the loader offers, floored at 1.2
        // (the device layer requires 1.2 + dynamic rendering)
        uint32_t loader_version = VK_API_VERSION_1_0;
        vkEnumerateInstanceVersion(&loader_version);
        if (loader_version < VK_API_VERSION_1_2)
            throw std::runtime_error("Vulkan loader below 1.2, cannot create instance");
        api_version_ = std::min(loader_version, VK_API_VERSION_1_3);

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "arenai";
        app_info.pEngineName = "arenai_view";
        app_info.apiVersion = api_version_;

        std::vector<const char *> layers;
        if (std::getenv("ARENAI_VK_VALIDATION") != nullptr && validation_layer_available())
            layers.push_back("VK_LAYER_KHRONOS_validation");

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
        create_info.ppEnabledLayerNames = layers.data();
        create_info.enabledExtensionCount = static_cast<uint32_t>(extra_extensions.size());
        create_info.ppEnabledExtensionNames = extra_extensions.data();

        vk_check(vkCreateInstance(&create_info, nullptr, &instance_), "vkCreateInstance");
    }

    VkInstance VulkanInstance::handle() const { return instance_; }

    uint32_t VulkanInstance::api_version() const { return api_version_; }

    VulkanInstance::~VulkanInstance() {
        if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    }

}// namespace arenai::view
