//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_RENDER_TARGET_H
#define ARENAI_VK_RENDER_TARGET_H

#include <memory>

#include "./device.h"
#include "./vk.h"
#include "./vma.h"

namespace arenai::view {

    // One render target image (color or depth) with its view, VMA-allocated.
    class Target {
    public:
        Target(
            std::shared_ptr<VulkanDevice> device, int width, int height, VkFormat format,
            VkImageUsageFlags usage, VkImageAspectFlags aspect,
            VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);

        Target(const Target &) = delete;
        Target &operator=(const Target &) = delete;

        VkImage image() const;
        VkImageView view() const;
        VkFormat format() const;
        int width() const;
        int height() const;

        ~Target();

    private:
        std::shared_ptr<VulkanDevice> device_;
        VkImage image_;
        VmaAllocation allocation_;
        VkImageView view_;
        VkFormat format_;
        int width_, height_;
    };

    // Records a full-image layout transition barrier.
    void record_image_barrier(
        VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout old_layout,
        VkImageLayout new_layout, VkAccessFlags src_access, VkAccessFlags dst_access,
        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage);

}// namespace arenai::view

#endif// ARENAI_VK_RENDER_TARGET_H
