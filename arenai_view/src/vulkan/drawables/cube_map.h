//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_CUBE_MAP_H
#define ARENAI_VK_CUBE_MAP_H

#include <filesystem>
#include <memory>

#include <glm/glm.hpp>

#include <arenai_utils/file_reader.h>

#include "../buffer.h"
#include "../texture.h"
#include "../vk.h"
#include "./vulkan_drawable.h"

namespace arenai::view {

    class VulkanCubeMap final : public VulkanDrawable {
    public:
        VulkanCubeMap(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            std::filesystem::path pngs_root_path);

        void draw(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos) override;

        ~VulkanCubeMap() override;

    private:
        void ensure_resources();

        std::shared_ptr<utils::AbstractResourceFileReader> file_reader_;
        std::filesystem::path pngs_root_path_;
        int nb_vertices_;

        std::unique_ptr<VulkanBuffer> vertices_;
        std::unique_ptr<VulkanTexture> texture_;
        VkDescriptorSetLayout texture_layout_ = VK_NULL_HANDLE;
        VkDescriptorSet texture_set_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline pipeline_ = VK_NULL_HANDLE;
    };

}// namespace arenai::view

#endif// ARENAI_VK_CUBE_MAP_H
