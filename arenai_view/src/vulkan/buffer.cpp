//
// Created by samuel on 17/07/2026.
//

#include "./buffer.h"

#include <cstring>
#include <utility>

#include "./errors.h"

namespace arenai::view {

    /*
     * VulkanBuffer (device local)
     */

    VulkanBuffer::VulkanBuffer(
        std::shared_ptr<VulkanDevice> device, const VkCommandPool pool, const void *data,
        const size_t size, const VkBufferUsageFlags usage)
        : device_(std::move(device)), buffer_(VK_NULL_HANDLE), allocation_(VK_NULL_HANDLE),
          size_(size) {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        vk_check(
            vmaCreateBuffer(
                device_->allocator(), &buffer_info, &alloc_info, &buffer_, &allocation_, nullptr),
            "vmaCreateBuffer (device local)");

        // staging copy
        VkBufferCreateInfo staging_info{};
        staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_info.size = size;
        staging_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo staging_alloc_info{};
        staging_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                                   | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation staging_allocation = VK_NULL_HANDLE;
        VmaAllocationInfo staging_mapped{};
        vk_check(
            vmaCreateBuffer(
                device_->allocator(), &staging_info, &staging_alloc_info, &staging,
                &staging_allocation, &staging_mapped),
            "vmaCreateBuffer (staging)");

        std::memcpy(staging_mapped.pMappedData, data, size);

        device_->immediate_submit(pool, [&](const VkCommandBuffer cmd) {
            VkBufferCopy copy{0, 0, size};
            vkCmdCopyBuffer(cmd, staging, buffer_, 1, &copy);
        });

        vmaDestroyBuffer(device_->allocator(), staging, staging_allocation);
    }

    VkBuffer VulkanBuffer::handle() const { return buffer_; }

    size_t VulkanBuffer::size() const { return size_; }

    VulkanBuffer::~VulkanBuffer() {
        if (buffer_ != VK_NULL_HANDLE) vmaDestroyBuffer(device_->allocator(), buffer_, allocation_);
    }

    /*
     * HostVisibleBuffer
     */

    HostVisibleBuffer::HostVisibleBuffer(
        std::shared_ptr<VulkanDevice> device, const size_t size, const VkBufferUsageFlags usage)
        : device_(std::move(device)), buffer_(VK_NULL_HANDLE), allocation_(VK_NULL_HANDLE),
          mapped_(nullptr), size_(size) {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_info.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo mapped{};
        vk_check(
            vmaCreateBuffer(
                device_->allocator(), &buffer_info, &alloc_info, &buffer_, &allocation_, &mapped),
            "vmaCreateBuffer (host visible)");
        mapped_ = mapped.pMappedData;
    }

    VkBuffer HostVisibleBuffer::handle() const { return buffer_; }

    void *HostVisibleBuffer::data() const { return mapped_; }

    size_t HostVisibleBuffer::size() const { return size_; }

    void HostVisibleBuffer::flush() const {
        vmaFlushAllocation(device_->allocator(), allocation_, 0, VK_WHOLE_SIZE);
    }

    void HostVisibleBuffer::invalidate() const {
        vmaInvalidateAllocation(device_->allocator(), allocation_, 0, VK_WHOLE_SIZE);
    }

    HostVisibleBuffer::~HostVisibleBuffer() {
        if (buffer_ != VK_NULL_HANDLE) vmaDestroyBuffer(device_->allocator(), buffer_, allocation_);
    }

}// namespace arenai::view
