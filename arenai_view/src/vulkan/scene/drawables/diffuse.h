//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_DIFFUSE_H
#define ARENAI_VK_DIFFUSE_H

#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include <arenai_utils/file_reader.h>

#include "../../core/buffer.h"
#include "../../core/vk.h"
#include "./shadow_drawable.h"

namespace arenai::view {

    // Flat-shaded matte drawable: the fragment shader derives per-face normals
    // from screen-space derivatives, so only vertex positions are uploaded.
    // Three pipelines, built lazily on first use so that renderers without
    // shadows (offscreen agent vision) pay nothing for the shadow ones.
    class VulkanDiffuse final : public VulkanShadowDrawable {
    public:
        VulkanDiffuse(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::vector<std::tuple<float, float, float>> &vertices, glm::vec4 color);

        void draw(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos) override;

        void draw_depth(const glm::mat4 &light_mvp_matrix) override;

        void draw_with_shadow(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos, glm::vec4 world_up) override;

        ~VulkanDiffuse() override;

    private:
        // vertex buffer + material UBO/set, shared by the three pipelines
        void ensure_common();
        void ensure_plain_pipeline();
        void ensure_depth_pipeline();
        void ensure_shadow_pipeline();
        void record_draw(
            VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet set0,
            uint32_t dynamic_offset_count, const glm::mat4 &mvp_matrix,
            const glm::mat4 &mv_matrix) const;

        std::vector<float> vbo_data_;
        glm::vec4 color_;
        int nb_vertices_;

        std::unique_ptr<VulkanBuffer> vertices_;
        std::unique_ptr<HostVisibleBuffer> material_;
        VkDescriptorSetLayout material_layout_ = VK_NULL_HANDLE;
        VkDescriptorSet material_set_ = VK_NULL_HANDLE;

        VkPipelineLayout plain_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline plain_pipeline_ = VK_NULL_HANDLE;
        VkPipelineLayout depth_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline depth_pipeline_ = VK_NULL_HANDLE;
        VkPipelineLayout shadow_pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline shadow_pipeline_ = VK_NULL_HANDLE;
    };

}// namespace arenai::view

#endif// ARENAI_VK_DIFFUSE_H
