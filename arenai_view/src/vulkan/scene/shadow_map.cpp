//
// Created by samuel on 17/07/2026.
//

#include "./shadow_map.h"

#include "../core/errors.h"

namespace arenai::view {

    VulkanShadowMap::VulkanShadowMap(const std::shared_ptr<VulkanDevice> &device, const int size)
        : device_(device), sampler_(VK_NULL_HANDLE), size_(size) {
        depth_ = std::make_unique<Target>(
            device, size, size, device->find_depth_format(true),
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.compareEnable = VK_TRUE;
        sampler_info.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        vk_check(
            vkCreateSampler(device_->handle(), &sampler_info, nullptr, &sampler_),
            "vkCreateSampler (shadow map)");
    }

    void VulkanShadowMap::begin_depth_pass(const VkCommandBuffer cmd) const {
        // the previous content is cleared below: UNDEFINED discards it and
        // covers the first use as well
        record_image_barrier(
            cmd, depth_->image(), VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

        VkRenderingAttachmentInfo depth_attachment{};
        depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_attachment.imageView = depth_->view();
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.clearValue.depthStencil = {1.f, 0};

        VkRenderingInfo rendering_info{};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.renderArea = {
            {0, 0}, {static_cast<uint32_t>(size_), static_cast<uint32_t>(size_)}};
        rendering_info.layerCount = 1;
        rendering_info.pDepthAttachment = &depth_attachment;

        vkCmdBeginRendering(cmd, &rendering_info);

        const VkViewport viewport{0.f,
                                  static_cast<float>(size_),
                                  static_cast<float>(size_),
                                  -static_cast<float>(size_),
                                  0.f,
                                  1.f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        const VkRect2D scissor{
            {0, 0}, {static_cast<uint32_t>(size_), static_cast<uint32_t>(size_)}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
    }

    void VulkanShadowMap::end_depth_pass(const VkCommandBuffer cmd) const {
        vkCmdEndRendering(cmd);

        record_image_barrier(
            cmd, depth_->image(), VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }

    VkImageView VulkanShadowMap::view() const { return depth_->view(); }

    VkSampler VulkanShadowMap::sampler() const { return sampler_; }

    VkFormat VulkanShadowMap::format() const { return depth_->format(); }

    int VulkanShadowMap::size() const { return size_; }

    VulkanShadowMap::~VulkanShadowMap() { vkDestroySampler(device_->handle(), sampler_, nullptr); }

}// namespace arenai::view
