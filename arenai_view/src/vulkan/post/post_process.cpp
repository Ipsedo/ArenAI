//
// Created by samuel on 17/07/2026.
//

#include "./post_process.h"

#include <utility>

#include "./ao_blur.h"
#include "./bloom_blur.h"
#include "./bloom_bright.h"
#include "./composite.h"
#include "./god_rays.h"
#include "./ssao.h"

namespace arenai::view {

    std::vector<std::shared_ptr<VulkanPostEffect>> make_default_post_processing_effects(
        const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
        const int width, const int height) {
        return {
            std::make_shared<SsaoEffect>(device, descriptors, width, height),
            std::make_shared<AoBlurEffect>(device, descriptors, width, height),
            std::make_shared<BloomBrightEffect>(device, descriptors, width, height),
            std::make_shared<BloomBlurEffect>(device, descriptors, width, height),
            std::make_shared<GodRaysEffect>(device, descriptors, width, height),
            std::make_shared<CompositeEffect>(device, descriptors, width, height)};
    }

    VulkanPostProcess::VulkanPostProcess(
        std::shared_ptr<VulkanDevice> device, DescriptorAllocator *descriptors, const int width,
        const int height, std::vector<std::shared_ptr<VulkanPostEffect>> ordered_effects)
        : device_(std::move(device)), descriptors_(descriptors), width_(width), height_(height),
          frame_(0), samples_(device_->clamp_sample_count(MSAA_SAMPLES)),
          depth_format_(device_->find_depth_format(true)),
          ordered_effects_(std::move(ordered_effects)) {
        create_scene_targets();
    }

    void VulkanPostProcess::create_scene_targets() {
        // single-sampled resolve targets, sampled by the effect chain
        resolve_color_ = std::make_unique<Target>(
            device_, width_, height_, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        resolve_depth_ = std::make_unique<Target>(
            device_, width_, height_, depth_format_,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);

        if (samples_ != VK_SAMPLE_COUNT_1_BIT) {
            msaa_color_ = std::make_unique<Target>(
                device_, width_, height_, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, samples_);
            msaa_depth_ = std::make_unique<Target>(
                device_, width_, height_, depth_format_,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, samples_);
        } else {
            // no MSAA available: render straight into the resolve targets
            msaa_color_.reset();
            msaa_depth_.reset();
        }
    }

    void VulkanPostProcess::resize(const int new_width, const int new_height) {
        if (new_width == width_ && new_height == height_) return;

        width_ = new_width;
        height_ = new_height;

        device_->wait_idle();
        create_scene_targets();

        for (const auto &effect: ordered_effects_) effect->resize(width_, height_);
    }

    VkFormat VulkanPostProcess::scene_color_format() const { return VK_FORMAT_R8G8B8A8_UNORM; }

    VkFormat VulkanPostProcess::scene_depth_format() const { return depth_format_; }

    VkSampleCountFlagBits VulkanPostProcess::scene_samples() const { return samples_; }

    void VulkanPostProcess::begin_scene_pass(const VkCommandBuffer cmd) {
        const bool msaa = samples_ != VK_SAMPLE_COUNT_1_BIT;
        const Target &color = msaa ? *msaa_color_ : *resolve_color_;
        const Target &depth = msaa ? *msaa_depth_ : *resolve_depth_;

        // the whole frame is redrawn: discard previous content; src stages
        // order against last frame's readers/writers of these images
        record_image_barrier(
            cmd, color.image(), VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        record_image_barrier(
            cmd, depth.image(), VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

        if (msaa) {
            record_image_barrier(
                cmd, resolve_color_->image(), VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
            record_image_barrier(
                cmd, resolve_depth_->image(), VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                    | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
        }

        VkRenderingAttachmentInfo color_attachment{};
        color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_attachment.imageView = color.view();
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp =
            msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
        // same red clear as the GL player renderer
        color_attachment.clearValue.color = {{1.f, 0.f, 0.f, 0.f}};
        if (msaa) {
            color_attachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            color_attachment.resolveImageView = resolve_color_->view();
            color_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        VkRenderingAttachmentInfo depth_attachment{};
        depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_attachment.imageView = depth.view();
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp =
            msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.clearValue.depthStencil = {1.f, 0};
        if (msaa) {
            // GL's depth blit takes one sample: SAMPLE_ZERO, mandated by the
            // 1.2 depth-stencil-resolve support
            depth_attachment.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
            depth_attachment.resolveImageView = resolve_depth_->view();
            depth_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }

        VkRenderingInfo rendering_info{};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.renderArea = {
            {0, 0}, {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)}};
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_attachment;
        rendering_info.pDepthAttachment = &depth_attachment;

        vkCmdBeginRendering(cmd, &rendering_info);

        // negative height: the scene keeps the GL orientation conventions
        const VkViewport viewport{0.f,
                                  static_cast<float>(height_),
                                  static_cast<float>(width_),
                                  -static_cast<float>(height_),
                                  0.f,
                                  1.f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        const VkRect2D scissor{
            {0, 0}, {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
    }

    void VulkanPostProcess::run_effects(
        const VkCommandBuffer cmd, const glm::mat4 &proj_matrix, const glm::vec3 &sun_dir_view) {
        vkCmdEndRendering(cmd);

        // the resolved color + depth become the inputs of the effect chain
        record_image_barrier(
            cmd, resolve_color_->image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        record_image_barrier(
            cmd, resolve_depth_->image(), VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        frame_ = (frame_ + 1) % 1024;

        context_ = VulkanPostEffect::FrameContext{
            cmd,
            resolve_color_.get(),
            resolve_depth_.get(),
            width_,
            height_,
            proj_matrix,
            // projection terms consumed by the depth-reconstruction shaders
            glm::vec4(proj_matrix[0][0], proj_matrix[1][1], proj_matrix[2][2], proj_matrix[3][2]),
            sun_dir_view,
            frame_,
            {},
            {},
            VK_FORMAT_UNDEFINED,
            0,
            0};

        for (size_t i = 0; i + 1 < ordered_effects_.size(); i++)
            ordered_effects_[i]->render(context_);
    }

    void VulkanPostProcess::composite_within(
        const VkCommandBuffer cmd, const VkFormat output_format, const int output_width,
        const int output_height) {
        context_.cmd = cmd;
        context_.output_format = output_format;
        context_.output_width = output_width;
        context_.output_height = output_height;
        ordered_effects_.back()->render(context_);
    }

}// namespace arenai::view
