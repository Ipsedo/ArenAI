//
// Created by samuel on 17/07/2026.
//

#include "./device.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "./errors.h"

namespace arenai::view {

    VulkanDevice::VulkanDevice(
        std::shared_ptr<VulkanInstance> instance, const DeviceCriteria &criteria)
        : instance_(std::move(instance)), physical_(VK_NULL_HANDLE), properties_(),
          queue_family_(0), device_(VK_NULL_HANDLE), queue_(VK_NULL_HANDLE),
          allocator_(VK_NULL_HANDLE), pipeline_cache_(VK_NULL_HANDLE), wide_lines_(false) {
        const auto [physical, properties, queue_family] =
            pick_physical_device(instance_->handle(), criteria);
        physical_ = physical;
        properties_ = properties;
        queue_family_ = queue_family;

        constexpr float priority = 1.f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;

        std::vector<const char *> extensions;
        if (criteria.surface != VK_NULL_HANDLE)
            extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

        // dynamic rendering: core in 1.3, extension + feature struct on 1.2
        VkPhysicalDeviceVulkan13Features features13{};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        features13.dynamicRendering = VK_TRUE;
        VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering{};
        dynamic_rendering.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
        dynamic_rendering.dynamicRendering = VK_TRUE;

        // wide lines (HUD line drawing) are optional: enabled when present
        VkPhysicalDeviceFeatures supported{};
        vkGetPhysicalDeviceFeatures(physical_, &supported);
        wide_lines_ = supported.wideLines == VK_TRUE;

        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.features.wideLines = wide_lines_ ? VK_TRUE : VK_FALSE;
        if (properties_.apiVersion >= VK_API_VERSION_1_3) {
            features.pNext = &features13;
        } else {
            features.pNext = &dynamic_rendering;
            extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
        }

        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.pNext = &features;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        device_info.ppEnabledExtensionNames = extensions.data();

        vk_check(vkCreateDevice(physical_, &device_info, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

        VmaAllocatorCreateInfo allocator_info{};
        allocator_info.instance = instance_->handle();
        allocator_info.physicalDevice = physical_;
        allocator_info.device = device_;
        allocator_info.vulkanApiVersion =
            std::min(properties_.apiVersion, instance_->api_version());
        vk_check(vmaCreateAllocator(&allocator_info, &allocator_), "vmaCreateAllocator");

        VkPipelineCacheCreateInfo cache_info{};
        cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        vk_check(
            vkCreatePipelineCache(device_, &cache_info, nullptr, &pipeline_cache_),
            "vkCreatePipelineCache");
    }

    VkDevice VulkanDevice::handle() const { return device_; }

    VkPhysicalDevice VulkanDevice::physical() const { return physical_; }

    uint32_t VulkanDevice::queue_family() const { return queue_family_; }

    VmaAllocator VulkanDevice::allocator() const { return allocator_; }

    VkPipelineCache VulkanDevice::pipeline_cache() const { return pipeline_cache_; }

    const VkPhysicalDeviceProperties &VulkanDevice::properties() const { return properties_; }

    void VulkanDevice::submit(const VkCommandBuffer cmd, const VkFence fence) {
        submit(cmd, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, fence);
    }

    void VulkanDevice::submit(
        const VkCommandBuffer cmd, const VkSemaphore wait, const VkPipelineStageFlags wait_stage,
        const VkSemaphore signal, const VkFence fence) {
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        if (wait != VK_NULL_HANDLE) {
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = &wait;
            submit_info.pWaitDstStageMask = &wait_stage;
        }
        if (signal != VK_NULL_HANDLE) {
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = &signal;
        }

        std::lock_guard lock(queue_mutex_);
        vk_check(vkQueueSubmit(queue_, 1, &submit_info, fence), "vkQueueSubmit");
    }

    VkResult VulkanDevice::present(
        const VkSwapchainKHR swapchain, const uint32_t image_index, const VkSemaphore wait) {
        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = wait != VK_NULL_HANDLE ? 1 : 0;
        present_info.pWaitSemaphores = &wait;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain;
        present_info.pImageIndices = &image_index;

        std::lock_guard lock(queue_mutex_);
        return vkQueuePresentKHR(queue_, &present_info);
    }

    void VulkanDevice::immediate_submit(
        const VkCommandPool pool, const std::function<void(VkCommandBuffer)> &record) {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vk_check(
            vkAllocateCommandBuffers(device_, &alloc_info, &cmd),
            "vkAllocateCommandBuffers (immediate)");

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer (immediate)");
        record(cmd);
        vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer (immediate)");

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vk_check(vkCreateFence(device_, &fence_info, nullptr, &fence), "vkCreateFence (immediate)");

        submit(cmd, fence);

        vk_check(
            vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences (immediate)");
        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, pool, 1, &cmd);
    }

    void VulkanDevice::wait_idle() {
        std::lock_guard lock(queue_mutex_);
        vk_check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle");
    }

    VkCommandPool VulkanDevice::make_command_pool() const {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = queue_family_;
        VkCommandPool pool = VK_NULL_HANDLE;
        vk_check(vkCreateCommandPool(device_, &pool_info, nullptr, &pool), "vkCreateCommandPool");
        return pool;
    }

    VkFormat VulkanDevice::find_depth_format(const bool needs_sampling) const {
        VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (needs_sampling) needed |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

        for (const auto format:
             {VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT}) {
            VkFormatProperties properties;
            vkGetPhysicalDeviceFormatProperties(physical_, format, &properties);
            if ((properties.optimalTilingFeatures & needed) == needed) return format;
        }
        throw std::runtime_error("no depth format with the required features");
    }

    VkSampleCountFlagBits VulkanDevice::clamp_sample_count(const int wanted) const {
        const VkSampleCountFlags supported = properties_.limits.framebufferColorSampleCounts
                                             & properties_.limits.framebufferDepthSampleCounts;
        int samples = wanted;
        while (samples > 1 && !(supported & samples)) samples /= 2;
        return static_cast<VkSampleCountFlagBits>(std::max(samples, 1));
    }

    bool VulkanDevice::wide_lines() const { return wide_lines_; }

    std::string VulkanDevice::renderer_info() const {
        const auto type = [&] {
            switch (properties_.deviceType) {
                case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
                case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
                case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
                case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
                default: return "other";
            }
        }();
        const auto version = [](const uint32_t v) {
            return std::to_string(VK_API_VERSION_MAJOR(v)) + "."
                   + std::to_string(VK_API_VERSION_MINOR(v)) + "."
                   + std::to_string(VK_API_VERSION_PATCH(v));
        };
        return "device=" + std::string(properties_.deviceName) + ", type=" + type
               + ", api=" + version(properties_.apiVersion)
               + ", driver=" + std::to_string(properties_.driverVersion);
    }

    VulkanDevice::~VulkanDevice() {
        if (device_ == VK_NULL_HANDLE) return;
        vkDeviceWaitIdle(device_);
        vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
        vmaDestroyAllocator(allocator_);
        vkDestroyDevice(device_, nullptr);
    }

}// namespace arenai::view
