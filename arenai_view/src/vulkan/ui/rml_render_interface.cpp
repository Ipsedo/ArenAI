//
// Created by samuel on 17/07/2026.
//

#include "./rml_render_interface.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

#include "../core/pipeline.h"

namespace arenai::view {

    namespace {
        struct RmlPush {
            glm::mat4 projection;
            glm::vec2 translation;
        };
    }// namespace

    RmlVulkanRenderInterface::RmlVulkanRenderInterface(
        std::shared_ptr<VulkanDevice> device, std::shared_ptr<WindowFrameContext> frame_context)
        : device_(std::move(device)), frame_context_(std::move(frame_context)) {}

    void RmlVulkanRenderInterface::lazy_init() {
        if (initialized_) return;

        upload_pool_ = device_->make_command_pool();
        descriptors_ = std::make_unique<DescriptorAllocator>(device_);

        texture_layout_ =
            DescriptorLayoutBuilder()
                .add_binding(
                    0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
                .build(device_->handle());

        pipeline_layout_ = make_pipeline_layout(
            device_->handle(), {texture_layout_},
            {{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(RmlPush)}});

        pipeline_ = PipelineBuilder()
                        .shaders("rml_vs.glsl", "rml_fs.glsl")
                        .vertex_input(
                            {{0, sizeof(Rml::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}},
                            {{0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Rml::Vertex, position)},
                             {1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(Rml::Vertex, colour)},
                             {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Rml::Vertex, tex_coord)}})
                        .cull_mode(VK_CULL_MODE_NONE)
                        .depth(false, false)
                        // RmlUi outputs premultiplied-alpha colors
                        .blend_premultiplied()
                        .color_format(frame_context_->swapchain_format())
                        .build(device_, pipeline_layout_);

        white_texture_ = std::make_unique<RmlTexture>();
        white_texture_->texture = VulkanTexture::make_white(device_, upload_pool_);
        white_texture_->set = descriptors_->allocate(texture_layout_);
        write_image_descriptor(
            device_->handle(), white_texture_->set, 0, white_texture_->texture->sampler(),
            white_texture_->texture->view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        initialized_ = true;
    }

    RmlVulkanRenderInterface::~RmlVulkanRenderInterface() {
        device_->wait_idle();
        retired_geometries_.drain_all();
        retired_textures_.drain_all();
        white_texture_.reset();
        if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_->handle(), pipeline_, nullptr);
        if (pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_->handle(), pipeline_layout_, nullptr);
        if (texture_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_->handle(), texture_layout_, nullptr);
        descriptors_.reset();
        if (upload_pool_ != VK_NULL_HANDLE)
            vkDestroyCommandPool(device_->handle(), upload_pool_, nullptr);
    }

    void
    RmlVulkanRenderInterface::begin_frame(const int viewport_width, const int viewport_height) {
        lazy_init();

        retired_geometries_.tick();
        retired_textures_.tick();

        viewport_width_ = viewport_width;
        viewport_height_ = viewport_height;

        // pixel coordinates with the origin at the top-left, as RmlUi expects
        projection_ = glm::ortho(
            0.f, static_cast<float>(viewport_width), static_cast<float>(viewport_height), 0.f);

        scissor_enabled_ = false;
        in_frame_ = true;
    }

    void RmlVulkanRenderInterface::end_frame() {
        // restore the full-viewport scissor for whoever draws next in this scope
        if (in_frame_) set_scissor(0, 0, viewport_width_, viewport_height_);
        in_frame_ = false;
    }

    Rml::CompiledGeometryHandle RmlVulkanRenderInterface::CompileGeometry(
        const Rml::Span<const Rml::Vertex> vertices, const Rml::Span<const int> indices) {
        lazy_init();

        auto *geometry = new CompiledGeometry{};
        geometry->nb_indices = static_cast<uint32_t>(indices.size());
        geometry->vertices = std::make_unique<VulkanBuffer>(
            device_, upload_pool_, vertices.data(), vertices.size() * sizeof(Rml::Vertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        geometry->indices = std::make_unique<VulkanBuffer>(
            device_, upload_pool_, indices.data(), indices.size() * sizeof(int),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        return reinterpret_cast<Rml::CompiledGeometryHandle>(geometry);
    }

    void RmlVulkanRenderInterface::RenderGeometry(
        const Rml::CompiledGeometryHandle geometry, const Rml::Vector2f translation,
        const Rml::TextureHandle texture) {
        // minimized window: the UI frame was skipped entirely
        if (!in_frame_) return;

        const auto *compiled = reinterpret_cast<const CompiledGeometry *>(geometry);
        const auto *rml_texture =
            texture ? reinterpret_cast<const RmlTexture *>(texture) : white_texture_.get();

        const VkCommandBuffer cmd = frame_context_->cmd();

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &rml_texture->set, 0,
            nullptr);

        const RmlPush push{projection_, glm::vec2(translation.x, translation.y)};
        vkCmdPushConstants(
            cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(RmlPush), &push);

        const VkBuffer vertex_buffer = compiled->vertices->handle();
        constexpr VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
        vkCmdBindIndexBuffer(cmd, compiled->indices->handle(), 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(cmd, compiled->nb_indices, 1, 0, 0, 0);
    }

    void RmlVulkanRenderInterface::ReleaseGeometry(const Rml::CompiledGeometryHandle geometry) {
        auto *compiled = reinterpret_cast<CompiledGeometry *>(geometry);
        retired_geometries_.retire(std::unique_ptr<CompiledGeometry>(compiled));
    }

    Rml::TextureHandle RmlVulkanRenderInterface::LoadTexture(
        Rml::Vector2i &texture_dimensions, const Rml::String &source) {
        // no image decoding on the UI path yet: the menu documents are styled
        // with plain decorators (colors, borders) and need no texture files
        std::cerr << "RmlUi texture files are not supported (requested: " << source << ")"
                  << std::endl;
        (void) texture_dimensions;
        return 0;
    }

    Rml::TextureHandle RmlVulkanRenderInterface::GenerateTexture(
        const Rml::Span<const Rml::byte> source, const Rml::Vector2i source_dimensions) {
        lazy_init();

        auto *rml_texture = new RmlTexture{};
        rml_texture->texture = std::make_unique<VulkanTexture>(
            device_, upload_pool_, source_dimensions.x, source_dimensions.y, 4, source.data());
        rml_texture->set = descriptors_->allocate(texture_layout_);
        write_image_descriptor(
            device_->handle(), rml_texture->set, 0, rml_texture->texture->sampler(),
            rml_texture->texture->view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        return reinterpret_cast<Rml::TextureHandle>(rml_texture);
    }

    void RmlVulkanRenderInterface::ReleaseTexture(const Rml::TextureHandle texture) {
        auto *rml_texture = reinterpret_cast<RmlTexture *>(texture);
        retired_textures_.retire(std::unique_ptr<RmlTexture>(rml_texture));
    }

    void RmlVulkanRenderInterface::set_scissor(
        const int x, const int y, const int width, const int height) const {
        if (!frame_context_->frame_active()) return;
        const VkRect2D scissor{
            {std::max(0, x), std::max(0, y)},
            {static_cast<uint32_t>(std::max(0, width)),
             static_cast<uint32_t>(std::max(0, height))}};
        vkCmdSetScissor(frame_context_->cmd(), 0, 1, &scissor);
    }

    void RmlVulkanRenderInterface::EnableScissorRegion(const bool enable) {
        if (!in_frame_) return;
        scissor_enabled_ = enable;
        if (!enable) set_scissor(0, 0, viewport_width_, viewport_height_);
    }

    void RmlVulkanRenderInterface::SetScissorRegion(const Rml::Rectanglei region) {
        if (!in_frame_ || !scissor_enabled_) return;
        // RmlUi regions and Vulkan scissors share the top-left origin
        set_scissor(region.Left(), region.Top(), region.Width(), region.Height());
    }

}// namespace arenai::view
