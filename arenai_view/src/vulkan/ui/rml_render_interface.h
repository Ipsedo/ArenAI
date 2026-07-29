//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_RML_RENDER_INTERFACE_H
#define ARENAI_VK_RML_RENDER_INTERFACE_H

#include <memory>

#include <glm/glm.hpp>
#include <RmlUi/Core/RenderInterface.h>

#include "../core/buffer.h"
#include "../core/descriptors.h"
#include "../core/device.h"
#include "../core/retire_queue.h"
#include "../core/texture.h"
#include "../core/vk.h"
#include "../present/window_frame.h"

namespace arenai::view {

    // Vulkan adapter behind RmlUi's abstract render interface. This is the
    // only place where the UI library touches the Vulkan API: the rest of the
    // application only sees the forward-declared Rml::RenderInterface handed
    // out by AbstractWindowedGraphicBackend. Draws are recorded into the
    // frame command buffer of the shared WindowFrameContext, inside the
    // swapchain rendering scope opened by begin_ui_frame/begin_ui_overlay.
    class RmlVulkanRenderInterface final : public Rml::RenderInterface {
    public:
        RmlVulkanRenderInterface(
            std::shared_ptr<VulkanDevice> device,
            std::shared_ptr<WindowFrameContext> frame_context);
        ~RmlVulkanRenderInterface() override;

        // called by the backend around Rml::Context::Render(), inside the
        // swapchain rendering scope
        void begin_frame(int viewport_width, int viewport_height);
        void end_frame();

        Rml::CompiledGeometryHandle CompileGeometry(
            Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) override;
        void RenderGeometry(
            Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation,
            Rml::TextureHandle texture) override;
        void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;

        Rml::TextureHandle
        LoadTexture(Rml::Vector2i &texture_dimensions, const Rml::String &source) override;
        Rml::TextureHandle GenerateTexture(
            Rml::Span<const Rml::byte> source, Rml::Vector2i source_dimensions) override;
        void ReleaseTexture(Rml::TextureHandle texture) override;

        void EnableScissorRegion(bool enable) override;
        void SetScissorRegion(Rml::Rectanglei region) override;

    private:
        struct CompiledGeometry {
            std::unique_ptr<VulkanBuffer> vertices;
            std::unique_ptr<VulkanBuffer> indices;
            uint32_t nb_indices;
        };

        struct RmlTexture {
            std::unique_ptr<VulkanTexture> texture;
            VkDescriptorSet set;
        };

        // pipeline & co are created on first begin_frame(), when the
        // swapchain format is known
        void lazy_init();
        void set_scissor(int x, int y, int width, int height) const;

        std::shared_ptr<VulkanDevice> device_;
        std::shared_ptr<WindowFrameContext> frame_context_;

        bool initialized_ = false;
        VkCommandPool upload_pool_ = VK_NULL_HANDLE;
        std::unique_ptr<DescriptorAllocator> descriptors_;
        VkDescriptorSetLayout texture_layout_ = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
        VkPipeline pipeline_ = VK_NULL_HANDLE;

        // bound for untextured geometry so that one pipeline handles both cases
        std::unique_ptr<RmlTexture> white_texture_;

        // resources released by RmlUi may still be referenced by an in-flight
        // frame: destruction is deferred by two frames (or to the destructor,
        // for releases fired outside any frame e.g. during Rml::Shutdown)
        RetireQueue<CompiledGeometry> retired_geometries_{WindowFrameContext::FRAME_SLOTS};
        RetireQueue<RmlTexture> retired_textures_{WindowFrameContext::FRAME_SLOTS};

        bool in_frame_ = false;
        glm::mat4 projection_{1.f};
        int viewport_width_ = 0;
        int viewport_height_ = 0;
        bool scissor_enabled_ = false;
    };

}// namespace arenai::view

#endif// ARENAI_VK_RML_RENDER_INTERFACE_H
