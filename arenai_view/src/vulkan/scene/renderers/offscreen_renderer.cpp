//
// Created by samuel on 17/07/2026.
//

#include "./offscreen_renderer.h"

#include <cstring>

#include "../../core/errors.h"

using namespace arenai;

namespace arenai::view {

    VulkanOffscreenRenderer::VulkanOffscreenRenderer(
        const std::shared_ptr<VulkanDevice> &device, const int width, const int height,
        const glm::vec3 light_pos, const std::shared_ptr<AbstractCamera> &camera,
        const bool with_shadows)
        : VulkanRenderer(device, light_pos, camera, with_shadows), width_(width), height_(height),
          depth_format_(device->find_depth_format(false)) {
        color_ = std::make_unique<Target>(
            device, width, height, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        depth_ = std::make_unique<Target>(
            device, width, height, depth_format_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        const size_t readback_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = VulkanRenderer::upload_pool();
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        for (auto &slot: slots_) {
            vk_check(
                vkAllocateCommandBuffers(device->handle(), &alloc_info, &slot.cmd),
                "vkAllocateCommandBuffers (offscreen slot)");
            vk_check(
                vkCreateFence(device->handle(), &fence_info, nullptr, &slot.fence),
                "vkCreateFence (offscreen slot)");
            slot.readback = std::make_unique<HostVisibleBuffer>(
                device, readback_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }
    }

    std::pair<VkCommandBuffer, int> VulkanOffscreenRenderer::on_begin_frame() {
        auto &slot = slots_[slot_index_];
        if (slot.submitted) {
            vk_check(
                vkWaitForFences(device()->handle(), 1, &slot.fence, VK_TRUE, UINT64_MAX),
                "vkWaitForFences (offscreen slot)");
            vk_check(
                vkResetFences(device()->handle(), 1, &slot.fence),
                "vkResetFences (offscreen slot)");
            slot.submitted = false;
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(slot.cmd, &begin_info), "vkBeginCommandBuffer (offscreen)");

        return {slot.cmd, slot_index_};
    }

    void VulkanOffscreenRenderer::on_begin_scene_pass() {
        const VkCommandBuffer &cmd = scene_frame().cmd;

        record_image_barrier(
            cmd, color_->image(), VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        record_image_barrier(
            cmd, depth_->image(), VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

        VkRenderingAttachmentInfo color_attachment{};
        color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_attachment.imageView = color_->view();
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        color_attachment.clearValue.color = {{1.f, 0.f, 0.f, 0.f}};

        VkRenderingAttachmentInfo depth_attachment{};
        depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_attachment.imageView = depth_->view();
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.clearValue.depthStencil = {1.f, 0};

        VkRenderingInfo rendering_info{};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.renderArea = {
            {0, 0}, {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)}};
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_attachment;
        rendering_info.pDepthAttachment = &depth_attachment;

        vkCmdBeginRendering(cmd, &rendering_info);

        const VkViewport viewport{
            .x = 0.f,
            .y = static_cast<float>(height_),
            .width = static_cast<float>(width_),
            .height = -static_cast<float>(height_),
            .minDepth = 0.f,
            .maxDepth = 1.f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        const VkRect2D scissor{
            .offset = {.x = 0, .y = 0},
            .extent = {
                .width = static_cast<uint32_t>(width_), .height = static_cast<uint32_t>(height_)}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
    }

    void VulkanOffscreenRenderer::on_end_frame(const glm::mat4 &, const glm::mat4 &) {
        auto &[cmd, fence, readback, submitted] = slots_[slot_index_];

        vkCmdEndRendering(cmd);

        record_image_barrier(
            cmd, color_->image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy copy{};
        copy.imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1};
        copy.imageExtent = {
            .width = static_cast<uint32_t>(width_),
            .height = static_cast<uint32_t>(height_),
            .depth = 1};
        vkCmdCopyImageToBuffer(
            cmd, color_->image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback->handle(), 1,
            &copy);

        vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer (offscreen)");

        device()->submit(cmd, fence);
        submitted = true;
        slot_index_ = (slot_index_ + 1) % FRAME_SLOTS;
    }

    image<uint8_t> VulkanOffscreenRenderer::read_slot(const Slot &slot) const {
        vk_check(
            vkWaitForFences(device()->handle(), 1, &slot.fence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences (offscreen readback)");
        slot.readback->invalidate();

        const int hw = width_ * height_;
        auto frame = image(std::vector<uint8_t>(hw * 3));

        const auto *src = static_cast<const uint8_t *>(slot.readback->data());
        auto *dst = frame.pixels.data();
        for (int i = 0; i < hw; i++) {
            dst[0 * hw + i] = src[i * 4 + 0];
            dst[1 * hw + i] = src[i * 4 + 1];
            dst[2 * hw + i] = src[i * 4 + 2];
        }
        return frame;
    }

    image<uint8_t> VulkanOffscreenRenderer::draw_and_get_frame(
        const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
        draw(model_matrices);

        const auto &previous = slots_[slot_index_];
        if (!previous.submitted) {
            const int hw = width_ * height_;
            return image(std::vector<uint8_t>(hw * 3, 0));
        }
        return read_slot(previous);
    }

    int VulkanOffscreenRenderer::get_width() const { return width_; }

    int VulkanOffscreenRenderer::get_height() const { return height_; }

    VkFormat VulkanOffscreenRenderer::scene_color_format() const {
        return VK_FORMAT_R8G8B8A8_UNORM;
    }

    VkFormat VulkanOffscreenRenderer::scene_depth_format() const { return depth_format_; }

    VkSampleCountFlagBits VulkanOffscreenRenderer::scene_samples() const {
        return VK_SAMPLE_COUNT_1_BIT;
    }

    VulkanOffscreenRenderer::~VulkanOffscreenRenderer() {
        for (auto &slot: slots_)
            if (slot.submitted)
                vkWaitForFences(
                    VulkanRenderer::device()->handle(), 1, &slot.fence, VK_TRUE, UINT64_MAX);
        for (auto &slot: slots_) {
            vkDestroyFence(VulkanRenderer::device()->handle(), slot.fence, nullptr);
            slot.readback.reset();
        }
    }

}// namespace arenai::view
