//
// Created by samuel on 17/07/2026.
//

#include "./cube_map.h"

#include <utility>
#include <vector>

#include "../descriptors.h"
#include "../pipeline.h"
#include "../renderers/renderer.h"

using namespace arenai;

namespace arenai::view {

    namespace {
        const std::vector<float> CUBE_VERTICES{-1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  -1.f, -1.f,
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

    VulkanCubeMap::VulkanCubeMap(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        std::filesystem::path pngs_root_path)
        : file_reader_(file_reader), pngs_root_path_(std::move(pngs_root_path)),
          nb_vertices_(static_cast<int>(CUBE_VERTICES.size() / 3)) {}

    void VulkanCubeMap::ensure_resources() {
        if (pipeline_ != VK_NULL_HANDLE) return;

        vertices_ = std::make_unique<VulkanBuffer>(
            renderer_->device(), renderer_->upload_pool(), CUBE_VERTICES.data(),
            CUBE_VERTICES.size() * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        texture_ = VulkanTexture::cube_from_pngs(
            renderer_->device(), renderer_->upload_pool(), file_reader_, pngs_root_path_);

        texture_layout_ =
            DescriptorLayoutBuilder()
                .add_binding(
                    0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
                .build(renderer_->device()->handle());
        texture_set_ = renderer_->descriptors().allocate(texture_layout_);
        write_image_descriptor(
            renderer_->device()->handle(), texture_set_, 0, texture_->sampler(), texture_->view(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        pipeline_layout_ = make_pipeline_layout(
            renderer_->device()->handle(), {renderer_->set0_plain_layout(), texture_layout_},
            {{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)}});
        pipeline_ = PipelineBuilder()
                        .shaders("cube_vs.glsl", "cube_fs.glsl")
                        .vertex_input(
                            {{0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX}},
                            {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}})
                        .color_format(renderer_->scene_color_format())
                        .depth_format(renderer_->scene_depth_format())
                        .samples(renderer_->scene_samples())
                        .build(renderer_->device(), pipeline_layout_);
    }

    void VulkanCubeMap::draw(const glm::mat4 mvp_matrix, glm::mat4, glm::vec3, glm::vec3) {
        ensure_resources();

        const auto &frame = renderer_->scene_frame();

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        const VkDescriptorSet sets[2] = {frame.set0_plain, texture_set_};
        vkCmdBindDescriptorSets(
            frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 2, sets, 0, nullptr);

        vkCmdPushConstants(
            frame.cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4),
            &mvp_matrix);

        const VkBuffer vertex_buffer = vertices_->handle();
        constexpr VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertex_buffer, &offset);

        vkCmdDraw(frame.cmd, nb_vertices_, 1, 0, 0);
    }

    VulkanCubeMap::~VulkanCubeMap() {
        if (renderer_ == nullptr) return;
        const VkDevice device = renderer_->device()->handle();
        vkDestroyPipeline(device, pipeline_, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        vkDestroyDescriptorSetLayout(device, texture_layout_, nullptr);
    }

}// namespace arenai::view
