//
// Created by samuel on 17/07/2026.
//

#include "./pipeline.h"

#include "./errors.h"
#include "./shader_modules.h"

namespace arenai::view {

    VkPipelineLayout make_pipeline_layout(
        const VkDevice device, const std::vector<VkDescriptorSetLayout> &set_layouts,
        const std::vector<VkPushConstantRange> &push_ranges) {
        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
        layout_info.pSetLayouts = set_layouts.data();
        layout_info.pushConstantRangeCount = static_cast<uint32_t>(push_ranges.size());
        layout_info.pPushConstantRanges = push_ranges.data();

        VkPipelineLayout layout = VK_NULL_HANDLE;
        vk_check(
            vkCreatePipelineLayout(device, &layout_info, nullptr, &layout),
            "vkCreatePipelineLayout");
        return layout;
    }

    PipelineBuilder &
    PipelineBuilder::shaders(const std::string &vertex_name, const std::string &fragment_name) {
        vertex_name_ = vertex_name;
        fragment_name_ = fragment_name;
        return *this;
    }

    PipelineBuilder &PipelineBuilder::vertex_input(
        const std::vector<VkVertexInputBindingDescription> &bindings,
        const std::vector<VkVertexInputAttributeDescription> &attributes) {
        bindings_ = bindings;
        attributes_ = attributes;
        return *this;
    }

    PipelineBuilder &PipelineBuilder::topology(const VkPrimitiveTopology topology) {
        topology_ = topology;
        return *this;
    }

    PipelineBuilder &PipelineBuilder::dynamic_line_width() {
        dynamic_line_width_ = true;
        return *this;
    }

    PipelineBuilder &PipelineBuilder::cull_mode(const VkCullModeFlags mode) {
        cull_mode_ = mode;
        return *this;
    }

    PipelineBuilder &PipelineBuilder::depth(const bool test, const bool write) {
        depth_test_ = test;
        depth_write_ = write;
        return *this;
    }

    PipelineBuilder &
    PipelineBuilder::depth_bias(const float constant_factor, const float slope_factor) {
        depth_bias_ = true;
        depth_bias_constant_ = constant_factor;
        depth_bias_slope_ = slope_factor;
        return *this;
    }

    PipelineBuilder &PipelineBuilder::blend_premultiplied() {
        blend_ = true;
        blend_src_ = VK_BLEND_FACTOR_ONE;
        blend_dst_ = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        return *this;
    }

    PipelineBuilder &PipelineBuilder::blend_alpha() {
        blend_ = true;
        blend_src_ = VK_BLEND_FACTOR_SRC_ALPHA;
        blend_dst_ = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        return *this;
    }

    PipelineBuilder &PipelineBuilder::color_format(const VkFormat format) {
        color_format_ = format;
        return *this;
    }

    PipelineBuilder &PipelineBuilder::depth_format(const VkFormat format) {
        depth_format_ = format;
        return *this;
    }

    PipelineBuilder &PipelineBuilder::samples(const VkSampleCountFlagBits samples) {
        samples_ = samples;
        return *this;
    }

    VkPipeline PipelineBuilder::build(
        const std::shared_ptr<VulkanDevice> &device, const VkPipelineLayout layout) const {
        const VkShaderModule vertex_module = load_shader_module(device->handle(), vertex_name_);
        const VkShaderModule fragment_module = load_shader_module(device->handle(), fragment_name_);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertex_module;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragment_module;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertex_input{};
        vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings_.size());
        vertex_input.pVertexBindingDescriptions = bindings_.data();
        vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes_.size());
        vertex_input.pVertexAttributeDescriptions = attributes_.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = topology_;

        VkPipelineViewportStateCreateInfo viewport{};
        viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterization{};
        rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = cull_mode_;
        // the negative-viewport convention keeps the GL winding: CCW = front
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.f;
        rasterization.depthBiasEnable = depth_bias_ ? VK_TRUE : VK_FALSE;
        rasterization.depthBiasConstantFactor = depth_bias_constant_;
        rasterization.depthBiasSlopeFactor = depth_bias_slope_;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = samples_;

        VkPipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable = depth_test_ ? VK_TRUE : VK_FALSE;
        depth_stencil.depthWriteEnable = depth_write_ ? VK_TRUE : VK_FALSE;
        depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState blend_attachment{};
        blend_attachment.blendEnable = blend_ ? VK_TRUE : VK_FALSE;
        blend_attachment.srcColorBlendFactor = blend_src_;
        blend_attachment.dstColorBlendFactor = blend_dst_;
        blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
        blend_attachment.srcAlphaBlendFactor = blend_src_;
        blend_attachment.dstAlphaBlendFactor = blend_dst_;
        blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
        blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                          | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo color_blend{};
        color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend.attachmentCount = color_format_ != VK_FORMAT_UNDEFINED ? 1 : 0;
        color_blend.pAttachments = &blend_attachment;

        VkDynamicState dynamic_states[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = 2;
        if (dynamic_line_width_)
            dynamic_states[dynamic.dynamicStateCount++] = VK_DYNAMIC_STATE_LINE_WIDTH;
        dynamic.pDynamicStates = dynamic_states;

        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = color_format_ != VK_FORMAT_UNDEFINED ? 1 : 0;
        rendering.pColorAttachmentFormats = &color_format_;
        rendering.depthAttachmentFormat = depth_format_;

        VkGraphicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.pNext = &rendering;
        pipeline_info.stageCount = 2;
        pipeline_info.pStages = stages;
        pipeline_info.pVertexInputState = &vertex_input;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport;
        pipeline_info.pRasterizationState = &rasterization;
        pipeline_info.pMultisampleState = &multisample;
        pipeline_info.pDepthStencilState = &depth_stencil;
        pipeline_info.pColorBlendState = &color_blend;
        pipeline_info.pDynamicState = &dynamic;
        pipeline_info.layout = layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult result = vkCreateGraphicsPipelines(
            device->handle(), device->pipeline_cache(), 1, &pipeline_info, nullptr, &pipeline);

        vkDestroyShaderModule(device->handle(), vertex_module, nullptr);
        vkDestroyShaderModule(device->handle(), fragment_module, nullptr);

        vk_check(result, "vkCreateGraphicsPipelines (" + vertex_name_ + "/" + fragment_name_ + ")");
        return pipeline;
    }

}// namespace arenai::view
