#ifndef ARENAI_VK_DIFFUSE_H
#define ARENAI_VK_DIFFUSE_H

#include <memory>

#include <glm/glm.hpp>

#include "../../core/buffer.h"
#include "../../core/vk.h"
#include "../scene_pipelines.h"
#include "./mesh.h"
#include "./shadow_drawable.h"

namespace arenai::view {

    class VulkanDiffuse final : public VulkanShadowDrawable {
    public:
        VulkanDiffuse(std::shared_ptr<const VulkanMesh> mesh, glm::vec4 color);

        void draw(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos) override;

        void draw_depth(const glm::mat4 &light_mvp_matrix) override;

        void draw_with_shadow(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos, glm::vec4 world_up) override;

    private:
        void ensure_material();

        void record_draw(
            const ScenePipeline &pipeline, VkDescriptorSet frame_set,
            std::span<const uint32_t> dynamic_offsets, const glm::mat4 &mvp_matrix,
            const glm::mat4 &mv_matrix);

        std::shared_ptr<const VulkanMesh> mesh_;
        glm::vec4 color_;

        std::unique_ptr<HostVisibleBuffer> material_;
        VkDescriptorSet material_set_ = VK_NULL_HANDLE;
    };

}// namespace arenai::view

#endif// ARENAI_VK_DIFFUSE_H
