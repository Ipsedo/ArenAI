//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_BUFFER_H
#define ARENAI_VK_BUFFER_H

#include <cstddef>
#include <memory>

#include "./device.h"
#include "./vk.h"
#include "./vma.h"

namespace arenai::view {

    class VulkanBuffer {
    public:
        VulkanBuffer(
            std::shared_ptr<VulkanDevice> device, const VkCommandPool &pool, const void *data,
            size_t size, VkBufferUsageFlags usage);

        VulkanBuffer(const VulkanBuffer &) = delete;
        VulkanBuffer &operator=(const VulkanBuffer &) = delete;

        VkBuffer handle() const;
        size_t size() const;

        ~VulkanBuffer();

    private:
        std::shared_ptr<VulkanDevice> device_;
        VkBuffer buffer_;
        VmaAllocation allocation_;
        size_t size_;
    };

    class HostVisibleBuffer {
    public:
        HostVisibleBuffer(
            std::shared_ptr<VulkanDevice> device, size_t size, const VkBufferUsageFlags &usage);

        HostVisibleBuffer(const HostVisibleBuffer &) = delete;
        HostVisibleBuffer &operator=(const HostVisibleBuffer &) = delete;

        VkBuffer handle() const;
        void *data() const;
        size_t size() const;

        void flush() const;
        void invalidate() const;

        ~HostVisibleBuffer();

    private:
        std::shared_ptr<VulkanDevice> device_;
        VkBuffer buffer_;
        VmaAllocation allocation_;
        void *mapped_;
        size_t size_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_BUFFER_H
