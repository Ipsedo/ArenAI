#include "./cube_map.h"

#include <span>
#include <utility>
#include <vector>

#include "../../core/descriptors.h"

using namespace arenai;

namespace arenai::view {

    namespace {
        const std::vector CUBE_VERTICES{-1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  -1.f, -1.f,
                                        1.f,  -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f, 1.f,  -1.f,

                                        -1.f, -1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  -1.f,
                                        -1.f, 1.f,  -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f, 1.f,

                                        1.f,  -1.f, -1.f, 1.f,  -1.f, 1.f,  1.f,  1.f,  1.f,
                                        1.f,  1.f,  1.f,  1.f,  1.f,  -1.f, 1.f,  -1.f, -1.f,

                                        -1.f, -1.f, 1.f,  -1.f, 1.f,  1.f,  1.f,  1.f,  1.f,
                                        1.f,  1.f,  1.f,  1.f,  -1.f, 1.f,  -1.f, -1.f, 1.f,

                                        -1.f, 1.f,  -1.f, 1.f,  1.f,  -1.f, 1.f,  1.f,  1.f,
                                        1.f,  1.f,  1.f,  -1.f, 1.f,  1.f,  -1.f, 1.f,  -1.f,

                                        -1.f, -1.f, -1.f, -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f,
                                        1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  1.f,  -1.f, 1.f};
    }// namespace

    VulkanCubeMap::VulkanCubeMap(std::shared_ptr<const VulkanCubeTexture> texture)
        : texture_(std::move(texture)), nb_vertices_(static_cast<int>(CUBE_VERTICES.size() / 3)) {}

    void VulkanCubeMap::ensure_resources() {
        if (vertices_) return;

        vertices_ = std::make_unique<VulkanBuffer>(
            context_->device(), context_->upload_pool(), CUBE_VERTICES.data(),
            CUBE_VERTICES.size() * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        texture_set_ = context_->descriptors().allocate(context_->pipelines().sky_layout());
        texture_->write_descriptor(context_->device()->handle(), texture_set_, 0);
    }

    void VulkanCubeMap::draw(const glm::mat4 mvp_matrix, glm::mat4, glm::vec3, glm::vec3) {
        ensure_resources();

        const auto &frame = context_->scene_frame();
        const auto &pipeline = context_->pipelines().sky();

        const VkDescriptorSet sets[2] = {frame.set0_plain, texture_set_};
        pipeline.bind(frame.cmd, sets, {});
        pipeline.push(frame.cmd, std::as_bytes(std::span(&mvp_matrix, 1)));

        const VkBuffer &vertex_buffer = vertices_->handle();
        constexpr VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertex_buffer, &offset);

        vkCmdDraw(frame.cmd, nb_vertices_, 1, 0, 0);
    }

}// namespace arenai::view
