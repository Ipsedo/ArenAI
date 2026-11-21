#include "./instance.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
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

        bool instance_extension_available(const char *name) {
            uint32_t count = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> extensions(count);
            vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
            return std::ranges::any_of(extensions, [name](const VkExtensionProperties &extension) {
                return std::strcmp(extension.extensionName, name) == 0;
            });
        }
    }// namespace

    VulkanInstance::VulkanInstance(const std::vector<const char *> &extra_extensions)
        : instance_(VK_NULL_HANDLE), api_version_(VK_API_VERSION_1_2) {
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

        std::vector<const char *> extensions = extra_extensions;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        // exposes non-conformant drivers (MoltenVK) hidden by the loader otherwise
        if (instance_extension_available(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
        create_info.pApplicationInfo = &app_info;
        create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
        create_info.ppEnabledLayerNames = layers.data();
        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();

        vk_check(vkCreateInstance(&create_info, nullptr, &instance_), "vkCreateInstance");
    }

    VkInstance VulkanInstance::handle() const { return instance_; }

    uint32_t VulkanInstance::api_version() const { return api_version_; }

    VulkanInstance::~VulkanInstance() {
        if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    }

}// namespace arenai::view
