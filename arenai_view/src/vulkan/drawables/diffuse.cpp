//
// Created by samuel on 17/07/2026.
//

#include "./diffuse.h"

#include <cstring>

#include "../descriptors.h"
#include "../pipeline.h"
#include "../renderers/renderer.h"

using namespace arenai;

namespace arenai::view {

    namespace {
        struct ScenePush {
            glm::mat4 mvp_matrix;
            glm::mat4 mv_matrix;
        };

        const std::vector<VkVertexInputBindingDescription> POSITION_BINDING = {
            {0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX}};
        const std::vector<VkVertexInputAttributeDescription> POSITION_ATTRIBUTE = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
    }// namespace

    VulkanDiffuse::VulkanDiffuse(
        const std::shared_ptr<utils::AbstractResourceFileReader> &,
        const std::vector<std::tuple<float, float, float>> &vertices, const glm::vec4 color)
        : color_(color), nb_vertices_(static_cast<int>(vertices.size())) {
        for (const auto &[x, y, z]: vertices) {
            vbo_data_.push_back(x);
            vbo_data_.push_back(y);
            vbo_data_.push_back(z);
        }
    }

    void VulkanDiffuse::ensure_common() {
        if (vertices_) return;

        vertices_ = std::make_unique<VulkanBuffer>(
            renderer_->device(), renderer_->upload_pool(), vbo_data_.data(),
            vbo_data_.size() * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        material_ = std::make_unique<HostVisibleBuffer>(
            renderer_->device(), sizeof(glm::vec4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        std::memcpy(material_->data(), &color_, sizeof(glm::vec4));
        material_->flush();

        material_layout_ =
            DescriptorLayoutBuilder()
                .add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
                .build(renderer_->device()->handle());
        material_set_ = renderer_->descriptors().allocate(material_layout_);
        write_buffer_descriptor(
            renderer_->device()->handle(), material_set_, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            material_->handle(), 0, sizeof(glm::vec4));
    }

    void VulkanDiffuse::ensure_plain_pipeline() {
        if (plain_pipeline_ != VK_NULL_HANDLE) return;

        plain_pipeline_layout_ = make_pipeline_layout(
            renderer_->device()->handle(), {renderer_->set0_plain_layout(), material_layout_},
            {{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ScenePush)}});
        plain_pipeline_ = PipelineBuilder()
                              .shaders("diffuse_vs.glsl", "diffuse_fs.glsl")
                              .vertex_input(POSITION_BINDING, POSITION_ATTRIBUTE)
                              .color_format(renderer_->scene_color_format())
                              .depth_format(renderer_->scene_depth_format())
                              .samples(renderer_->scene_samples())
                              .build(renderer_->device(), plain_pipeline_layout_);
    }

    void VulkanDiffuse::ensure_depth_pipeline() {
        if (depth_pipeline_ != VK_NULL_HANDLE) return;

        depth_pipeline_layout_ = make_pipeline_layout(
            renderer_->device()->handle(), {},
            {{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)}});
        depth_pipeline_ = PipelineBuilder()
                              .shaders("shadow_depth_vs.glsl", "shadow_depth_fs.glsl")
                              .vertex_input(POSITION_BINDING, POSITION_ATTRIBUTE)
                              // slope-scaled acne removal, ex-glPolygonOffset(2, 4)
                              .depth_bias(4.f, 2.f)
                              .depth_format(renderer_->shadow_depth_format())
                              .build(renderer_->device(), depth_pipeline_layout_);
    }

    void VulkanDiffuse::ensure_shadow_pipeline() {
        if (shadow_pipeline_ != VK_NULL_HANDLE) return;

        shadow_pipeline_layout_ = make_pipeline_layout(
            renderer_->device()->handle(), {renderer_->set0_shadow_layout(), material_layout_},
            {{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ScenePush)}});
        shadow_pipeline_ = PipelineBuilder()
                               .shaders("diffuse_shadow_vs.glsl", "diffuse_shadow_fs.glsl")
                               .vertex_input(POSITION_BINDING, POSITION_ATTRIBUTE)
                               .color_format(renderer_->scene_color_format())
                               .depth_format(renderer_->scene_depth_format())
                               .samples(renderer_->scene_samples())
                               .build(renderer_->device(), shadow_pipeline_layout_);
    }

    void VulkanDiffuse::record_draw(
        const VkPipeline pipeline, const VkPipelineLayout layout, const VkDescriptorSet set0,
        const uint32_t dynamic_offset_count, const glm::mat4 &mvp_matrix,
        const glm::mat4 &mv_matrix) const {
        const auto &frame = renderer_->scene_frame();

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        const VkDescriptorSet sets[2] = {set0, material_set_};
        vkCmdBindDescriptorSets(
            frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 2, sets, dynamic_offset_count,
            &frame.shadow_dynamic_offset);

        const ScenePush push{mvp_matrix, mv_matrix};
        vkCmdPushConstants(
            frame.cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ScenePush), &push);

        const VkBuffer vertex_buffer = vertices_->handle();
        constexpr VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertex_buffer, &offset);

        vkCmdDraw(frame.cmd, nb_vertices_, 1, 0, 0);
    }

    void VulkanDiffuse::draw(
        const glm::mat4 mvp_matrix, const glm::mat4 mv_matrix, glm::vec3, glm::vec3) {
        ensure_common();
        ensure_plain_pipeline();
        record_draw(
            plain_pipeline_, plain_pipeline_layout_, renderer_->scene_frame().set0_plain, 0,
            mvp_matrix, mv_matrix);
    }

    void VulkanDiffuse::draw_depth(const glm::mat4 &light_mvp_matrix) {
        ensure_common();
        ensure_depth_pipeline();

        const auto &frame = renderer_->scene_frame();
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_pipeline_);
        vkCmdPushConstants(
            frame.cmd, depth_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
            &light_mvp_matrix);

        const VkBuffer vertex_buffer = vertices_->handle();
        constexpr VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertex_buffer, &offset);

        vkCmdDraw(frame.cmd, nb_vertices_, 1, 0, 0);
    }

    void VulkanDiffuse::draw_with_shadow(
        const glm::mat4 mvp_matrix, const glm::mat4 mv_matrix, glm::vec3, glm::vec3, glm::vec4) {
        ensure_common();
        ensure_shadow_pipeline();
        record_draw(
            shadow_pipeline_, shadow_pipeline_layout_, renderer_->scene_frame().set0_shadow, 1,
            mvp_matrix, mv_matrix);
    }

    VulkanDiffuse::~VulkanDiffuse() {
        if (renderer_ == nullptr) return;
        const VkDevice device = renderer_->device()->handle();
        vkDestroyPipeline(device, shadow_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, shadow_pipeline_layout_, nullptr);
        vkDestroyPipeline(device, depth_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, depth_pipeline_layout_, nullptr);
        vkDestroyPipeline(device, plain_pipeline_, nullptr);
        vkDestroyPipelineLayout(device, plain_pipeline_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, material_layout_, nullptr);
    }

}// namespace arenai::view
