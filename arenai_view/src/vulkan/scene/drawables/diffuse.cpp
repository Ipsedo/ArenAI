#include "./diffuse.h"

#include <cstring>
#include <utility>

#include "../../core/descriptors.h"

using namespace arenai;

namespace arenai::view {

    namespace {
        struct ScenePush {
            glm::mat4 mvp_matrix;
            glm::mat4 mv_matrix;
        };
    }// namespace

    VulkanDiffuse::VulkanDiffuse(std::shared_ptr<const VulkanMesh> mesh, const glm::vec4 color)
        : mesh_(std::move(mesh)), color_(color) {}

    void VulkanDiffuse::ensure_material() {
        if (material_) return;

        material_ = std::make_unique<HostVisibleBuffer>(
            context_->device(), sizeof(glm::vec4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        std::memcpy(material_->data(), &color_, sizeof(glm::vec4));
        material_->flush();

        material_set_ = context_->descriptors().allocate(context_->pipelines().material_layout());
        write_buffer_descriptor(
            context_->device()->handle(), material_set_, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            material_->handle(), 0, sizeof(glm::vec4));
    }

    void VulkanDiffuse::record_draw(
        const ScenePipeline &pipeline, const VkDescriptorSet frame_set,
        const std::span<const uint32_t> dynamic_offsets, const glm::mat4 &mvp_matrix,
        const glm::mat4 &mv_matrix) {
        ensure_material();
        const auto &frame = context_->scene_frame();

        const VkDescriptorSet sets[2] = {frame_set, material_set_};
        pipeline.bind(frame.cmd, sets, dynamic_offsets);

        const ScenePush push{.mvp_matrix = mvp_matrix, .mv_matrix = mv_matrix};
        pipeline.push(frame.cmd, std::as_bytes(std::span(&push, 1)));

        mesh_->record_draw(frame.cmd);
    }

    void VulkanDiffuse::draw(
        const glm::mat4 mvp_matrix, const glm::mat4 mv_matrix, glm::vec3, glm::vec3) {
        record_draw(
            context_->pipelines().diffuse(), context_->scene_frame().set0_plain, {}, mvp_matrix,
            mv_matrix);
    }

    void VulkanDiffuse::draw_depth(const glm::mat4 &light_mvp_matrix) {
        const auto &frame = context_->scene_frame();
        const auto &pipeline = context_->pipelines().diffuse_depth();

        pipeline.bind(frame.cmd, {}, {});
        pipeline.push(frame.cmd, std::as_bytes(std::span(&light_mvp_matrix, 1)));

        mesh_->record_draw(frame.cmd);
    }

    void VulkanDiffuse::draw_with_shadow(
        const glm::mat4 mvp_matrix, const glm::mat4 mv_matrix, glm::vec3, glm::vec3, glm::vec4) {
        const auto &frame = context_->scene_frame();
        record_draw(
            context_->pipelines().diffuse_shadowed(), frame.set0_shadow,
            std::span(&frame.shadow_dynamic_offset, 1), mvp_matrix, mv_matrix);
    }

}// namespace arenai::view
