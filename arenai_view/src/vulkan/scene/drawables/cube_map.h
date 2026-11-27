#ifndef ARENAI_VK_CUBE_MAP_H
#define ARENAI_VK_CUBE_MAP_H

#include <memory>

#include <glm/glm.hpp>

#include "../../core/buffer.h"
#include "../../core/vk.h"
#include "./cube_texture.h"
#include "./vulkan_drawable.h"

namespace arenai::view {

    class VulkanCubeMap final : public VulkanDrawable {
    public:
        explicit VulkanCubeMap(std::shared_ptr<const VulkanCubeTexture> texture);

        void draw(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos) override;

    private:
        void ensure_resources();

        std::shared_ptr<const VulkanCubeTexture> texture_;
        int nb_vertices_;

        std::unique_ptr<VulkanBuffer> vertices_;
        VkDescriptorSet texture_set_ = VK_NULL_HANDLE;
    };

}// namespace arenai::view

#endif// ARENAI_VK_CUBE_MAP_H
