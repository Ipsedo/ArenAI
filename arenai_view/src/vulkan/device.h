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
#include "./vk.h"
#include "./vma.h"

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

        // asynchronous submit; the fence (optional) signals completion
        void submit(VkCommandBuffer cmd, VkFence fence);
        // full form for the windowed frame loop (acquire/present semaphores)
        void submit(
            VkCommandBuffer cmd, VkSemaphore wait, VkPipelineStageFlags wait_stage,
            VkSemaphore signal, VkFence fence);
        // returns VK_ERROR_OUT_OF_DATE_KHR / VK_SUBOPTIMAL_KHR untouched so the
        // caller can recreate the swapchain
        VkResult present(VkSwapchainKHR swapchain, uint32_t image_index, VkSemaphore wait);
        // records with a one-shot command buffer from the caller's (thread
        // confined) pool, submits and waits: reset-time uploads only
        void
        immediate_submit(VkCommandPool pool, const std::function<void(VkCommandBuffer)> &record);
        void wait_idle();

        VkCommandPool make_command_pool() const;

        // D24 when available (parity with the GL depth buffers), D32F otherwise
        VkFormat find_depth_format(bool needs_sampling) const;
        VkSampleCountFlagBits clamp_sample_count(int wanted) const;
        // whether line widths above 1.0 are supported (HUD line drawing)
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
