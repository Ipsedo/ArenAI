//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_PIPELINE_H
#define ARENAI_VK_PIPELINE_H

#include <memory>
#include <string>
#include <vector>

#include "./device.h"
#include "./vk.h"

namespace arenai::view {

    VkPipelineLayout make_pipeline_layout(
        VkDevice device, const std::vector<VkDescriptorSetLayout> &set_layouts,
        const std::vector<VkPushConstantRange> &push_ranges);

    // Graphics pipeline builder for dynamic rendering (no render pass), with
    // dynamic viewport + scissor. Defaults: triangle list, back-face culling
    // (CCW front), depth LESS_OR_EQUAL test+write, no blending, 1 sample.
    class PipelineBuilder {
    public:
        // shader names as embedded at build time, e.g. "diffuse_vs.glsl"
        PipelineBuilder &shaders(const std::string &vertex_name, const std::string &fragment_name);
        PipelineBuilder &vertex_input(
            const std::vector<VkVertexInputBindingDescription> &bindings,
            const std::vector<VkVertexInputAttributeDescription> &attributes);
        PipelineBuilder &topology(VkPrimitiveTopology topology);
        // adds VK_DYNAMIC_STATE_LINE_WIDTH (line pipelines only)
        PipelineBuilder &dynamic_line_width();
        PipelineBuilder &cull_mode(VkCullModeFlags mode);
        PipelineBuilder &depth(bool test, bool write);
        PipelineBuilder &depth_bias(float constant_factor, float slope_factor);
        // src=ONE, dst=ONE_MINUS_SRC_ALPHA: premultiplied-alpha blending
        PipelineBuilder &blend_premultiplied();
        // src=SRC_ALPHA, dst=ONE_MINUS_SRC_ALPHA: classic alpha blending
        PipelineBuilder &blend_alpha();
        PipelineBuilder &color_format(VkFormat format);// none = depth-only pass
        PipelineBuilder &depth_format(VkFormat format);
        PipelineBuilder &samples(VkSampleCountFlagBits samples);

        VkPipeline
        build(const std::shared_ptr<VulkanDevice> &device, VkPipelineLayout layout) const;

    private:
        std::string vertex_name_;
        std::string fragment_name_;
        std::vector<VkVertexInputBindingDescription> bindings_;
        std::vector<VkVertexInputAttributeDescription> attributes_;
        VkPrimitiveTopology topology_ = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        bool dynamic_line_width_ = false;
        VkCullModeFlags cull_mode_ = VK_CULL_MODE_BACK_BIT;
        bool depth_test_ = true;
        bool depth_write_ = true;
        bool depth_bias_ = false;
        float depth_bias_constant_ = 0.f;
        float depth_bias_slope_ = 0.f;
        bool blend_ = false;
        VkBlendFactor blend_src_ = VK_BLEND_FACTOR_ONE;
        VkBlendFactor blend_dst_ = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        VkFormat color_format_ = VK_FORMAT_UNDEFINED;
        VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
        VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    };

}// namespace arenai::view

#endif// ARENAI_VK_PIPELINE_H
