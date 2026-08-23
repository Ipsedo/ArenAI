//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_DEVICE_H
#define ARENAI_VK_DEVICE_H

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "./instance.h"
#include "./physical_device_selection.h"
#include "./vk.h"
#include "./vma.h"

namespace arenai::view {

    // One logical device with a single graphics queue. The device is shared
    // by every renderer of a backend, across threads: command pools and
    // descriptor pools stay per-renderer (thread-confined), only the queue
    // itself is serialized here (submit/present/immediate_submit lock).
    class VulkanDevice {
    public:
        VulkanDevice(std::shared_ptr<VulkanInstance> instance, const DeviceCriteria &criteria);

        VulkanDevice(const VulkanDevice &) = delete;
        VulkanDevice &operator=(const VulkanDevice &) = delete;

        VkDevice handle() const;
        VkPhysicalDevice physical() const;
        uint32_t queue_family() const;
        VmaAllocator allocator() const;
        VkPipelineCache pipeline_cache() const;
        const VkPhysicalDeviceProperties &properties() const;

        void submit(const VkCommandBuffer &cmd, const VkFence &fence);
        void submit(
            const VkCommandBuffer &cmd, const VkSemaphore &wait, VkPipelineStageFlags wait_stage,
            const VkSemaphore &signal, const VkFence &fence);

        VkResult
        present(const VkSwapchainKHR &swap_chain, uint32_t image_index, const VkSemaphore &wait);

        void immediate_submit(
            const VkCommandPool &pool, const std::function<void(VkCommandBuffer)> &record);

        void wait_idle();

        VkCommandPool make_command_pool() const;

        VkFormat find_depth_format(bool needs_sampling) const;
        VkSampleCountFlagBits clamp_sample_count(int wanted) const;

        bool wide_lines() const;

        std::string renderer_info() const;

        ~VulkanDevice();

    private:
        std::shared_ptr<VulkanInstance> instance_;

        VkPhysicalDevice physical_;
        VkPhysicalDeviceProperties properties_;
        uint32_t queue_family_;

        VkDevice device_;
        VkQueue queue_;
        std::mutex queue_mutex_;

        VmaAllocator allocator_;
        VkPipelineCache pipeline_cache_;
        bool wide_lines_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_DEVICE_H
