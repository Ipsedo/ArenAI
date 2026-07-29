//
// Created by samuel on 17/07/2026.
//

#include "./render_target.h"

#include <utility>

#include "./errors.h"

namespace arenai::view {

    Target::Target(
        std::shared_ptr<VulkanDevice> device, const int width, const int height,
        const VkFormat format, const VkImageUsageFlags usage, const VkImageAspectFlags aspect,
        const VkSampleCountFlagBits samples)
        : device_(std::move(device)), image_(VK_NULL_HANDLE), allocation_(VK_NULL_HANDLE),
          view_(VK_NULL_HANDLE), format_(format), width_(width), height_(height) {
        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = format;
        image_info.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = samples;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = usage;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        // render targets (especially the 16k shadow map) deserve their own block
        alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

        vk_check(
            vmaCreateImage(
                device_->allocator(), &image_info, &alloc_info, &image_, &allocation_, nullptr),
            "vmaCreateImage (render target)");

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = image_;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format;
        view_info.subresourceRange = {aspect, 0, 1, 0, 1};

        vk_check(
            vkCreateImageView(device_->handle(), &view_info, nullptr, &view_),
            "vkCreateImageView (render target)");
    }

    VkImage Target::image() const { return image_; }

    VkImageView Target::view() const { return view_; }

    VkFormat Target::format() const { return format_; }

    int Target::width() const { return width_; }

    int Target::height() const { return height_; }

    Target::~Target() {
        if (view_ != VK_NULL_HANDLE) vkDestroyImageView(device_->handle(), view_, nullptr);
        if (image_ != VK_NULL_HANDLE) vmaDestroyImage(device_->allocator(), image_, allocation_);
    }

    void record_image_barrier(
        const VkCommandBuffer cmd, const VkImage image, const VkImageAspectFlags aspect,
        const VkImageLayout old_layout, const VkImageLayout new_layout,
        const VkAccessFlags src_access, const VkAccessFlags dst_access,
        const VkPipelineStageFlags src_stage, const VkPipelineStageFlags dst_stage) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {
            aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        barrier.srcAccessMask = src_access;
        barrier.dstAccessMask = dst_access;

        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

}// namespace arenai::view
