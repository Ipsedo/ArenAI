//
// Created by samuel on 17/07/2026.
//

#include "./effect.h"

#include <algorithm>
#include <utility>

#include "../errors.h"
#include "../pipeline.h"

namespace arenai::view {

    namespace {
        bool is_depth_format(const VkFormat format) {
            return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_X8_D24_UNORM_PACK32
                   || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM;
        }
    }// namespace

    VulkanPostEffect::VulkanPostEffect(
        std::shared_ptr<VulkanDevice> device, DescriptorAllocator *descriptors,
        std::string fragment_shader, const uint32_t nb_inputs, const uint32_t push_size,
        std::vector<TargetSpec> specs, const int width, const int height)
        : device_(std::move(device)), descriptors_(descriptors),
          fragment_shader_(std::move(fragment_shader)), nb_inputs_(nb_inputs),
          push_size_(push_size), specs_(std::move(specs)) {
        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vk_check(
            vkCreateSampler(device_->handle(), &sampler_info, nullptr, &linear_sampler_),
            "vkCreateSampler (effect linear)");
        // depth inputs must be sampled NEAREST, like the GL resolve texture
        sampler_info.magFilter = VK_FILTER_NEAREST;
        sampler_info.minFilter = VK_FILTER_NEAREST;
        vk_check(
            vkCreateSampler(device_->handle(), &sampler_info, nullptr, &nearest_sampler_),
            "vkCreateSampler (effect nearest)");

        DescriptorLayoutBuilder layout_builder;
        for (uint32_t binding = 0; binding < nb_inputs_; binding++)
            layout_builder.add_binding(
                binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT);
        input_layout_ = layout_builder.build(device_->handle());
        // the effect shaders bind their samplers on set 1: set 0 stays an
        // empty placeholder, never bound
        empty_layout_ = DescriptorLayoutBuilder().build(device_->handle());

        std::vector<VkPushConstantRange> push_ranges;
        if (push_size_ > 0) push_ranges.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, 0, push_size_});
        pipeline_layout_ =
            make_pipeline_layout(device_->handle(), {empty_layout_, input_layout_}, push_ranges);

        create_targets(width, height);
    }

    void VulkanPostEffect::create_targets(const int width, const int height) {
        for (const auto &[format, size_divisor]: specs_)
            targets_.push_back(std::make_unique<Target>(
                device_, std::max(1, width / size_divisor), std::max(1, height / size_divisor),
                format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT));
        target_initialized_.assign(targets_.size(), false);
    }

    void VulkanPostEffect::resize(const int new_width, const int new_height) {
        device_->wait_idle();
        targets_.clear();
        // recreated targets may reuse freed addresses: drop the cached sets
        input_sets_.clear();
        create_targets(new_width, new_height);
    }

    VkPipeline VulkanPostEffect::pipeline_for(const VkFormat color_format) {
        const auto cached = pipelines_.find(color_format);
        if (cached != pipelines_.end()) return cached->second;

        const VkPipeline pipeline = PipelineBuilder()
                                        .shaders("post_vs.glsl", fragment_shader_)
                                        .cull_mode(VK_CULL_MODE_NONE)
                                        .depth(false, false)
                                        .color_format(color_format)
                                        .build(device_, pipeline_layout_);
        pipelines_.insert({color_format, pipeline});
        return pipeline;
    }

    VkDescriptorSet VulkanPostEffect::set_for(const std::vector<const Target *> &inputs) {
        const auto cached = input_sets_.find(inputs);
        if (cached != input_sets_.end()) return cached->second;

        const VkDescriptorSet set = descriptors_->allocate(input_layout_);
        for (uint32_t binding = 0; binding < inputs.size(); binding++) {
            const auto *input = inputs[binding];
            const bool depth = is_depth_format(input->format());
            write_image_descriptor(
                device_->handle(), set, binding, depth ? nearest_sampler_ : linear_sampler_,
                input->view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        input_sets_.insert({inputs, set});
        return set;
    }

    void VulkanPostEffect::record_draw(
        const FrameContext &context, const VkFormat color_format,
        const std::vector<const Target *> &inputs, const void *push_data) {
        const VkCommandBuffer cmd = context.cmd;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_for(color_format));

        const VkDescriptorSet set = set_for(inputs);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 1, 1, &set, 0, nullptr);

        if (push_size_ > 0 && push_data != nullptr)
            vkCmdPushConstants(
                cmd, pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, push_size_, push_data);

        // buffer-less fullscreen triangle (see post_vs.glsl)
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    void VulkanPostEffect::run_pass(
        const FrameContext &context, const size_t target_index,
        const std::vector<const Target *> &inputs, const void *push_data) {
        const VkCommandBuffer cmd = context.cmd;
        const Target &out = *targets_[target_index];

        // previous readers of this target are done before it is rewritten
        record_image_barrier(
            cmd, out.image(), VK_IMAGE_ASPECT_COLOR_BIT,
            target_initialized_[target_index] ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                              : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        VkRenderingAttachmentInfo color_attachment{};
        color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_attachment.imageView = out.view();
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo rendering_info{};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.renderArea = {
            {0, 0}, {static_cast<uint32_t>(out.width()), static_cast<uint32_t>(out.height())}};
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_attachment;

        vkCmdBeginRendering(cmd, &rendering_info);

        // regular viewport: image-space pass, rows map 1:1
        const VkViewport viewport{
            0.f, 0.f, static_cast<float>(out.width()), static_cast<float>(out.height()), 0.f, 1.f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        const VkRect2D scissor{
            {0, 0}, {static_cast<uint32_t>(out.width()), static_cast<uint32_t>(out.height())}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        record_draw(context, out.format(), inputs, push_data);

        vkCmdEndRendering(cmd);

        record_image_barrier(
            cmd, out.image(), VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        target_initialized_[target_index] = true;
    }

    void VulkanPostEffect::run_inline(
        const FrameContext &context, const std::vector<const Target *> &inputs,
        const void *push_data) {
        // image-space viewport inside the caller's scope (which set the
        // negative scene viewport): rows map 1:1 onto the output
        const VkViewport viewport{0.f,
                                  0.f,
                                  static_cast<float>(context.output_width),
                                  static_cast<float>(context.output_height),
                                  0.f,
                                  1.f};
        vkCmdSetViewport(context.cmd, 0, 1, &viewport);

        record_draw(context, context.output_format, inputs, push_data);
    }

    void VulkanPostEffect::ensure_target_readable(
        const FrameContext &context, const size_t target_index) {
        if (target_initialized_[target_index]) return;
        // content stays undefined (the consumer weights it to zero): only the
        // layout has to be valid for sampling
        record_image_barrier(
            context.cmd, targets_[target_index]->image(), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        target_initialized_[target_index] = true;
    }

    const Target *VulkanPostEffect::target(const size_t index) const {
        return targets_[index].get();
    }

    VulkanPostEffect::~VulkanPostEffect() {
        for (const auto &[format, pipeline]: pipelines_)
            vkDestroyPipeline(device_->handle(), pipeline, nullptr);
        vkDestroyPipelineLayout(device_->handle(), pipeline_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device_->handle(), input_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device_->handle(), empty_layout_, nullptr);
        vkDestroySampler(device_->handle(), nearest_sampler_, nullptr);
        vkDestroySampler(device_->handle(), linear_sampler_, nullptr);
    }

}// namespace arenai::view
