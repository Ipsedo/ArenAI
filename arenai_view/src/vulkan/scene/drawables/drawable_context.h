//
// Created by samuel on 20/07/2026.
//

#ifndef ARENAI_VK_DRAWABLE_CONTEXT_H
#define ARENAI_VK_DRAWABLE_CONTEXT_H

#include <memory>

#include "../../core/descriptors.h"
#include "../../core/device.h"
#include "../../core/vk.h"

namespace arenai::view {

    // Snapshot of the scene frame being recorded, shared with every drawable
    // draw call of that frame.
    struct SceneFrame {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        int slot = 0;
        VkDescriptorSet set0_plain = VK_NULL_HANDLE;
        VkDescriptorSet set0_shadow = VK_NULL_HANDLE;
        // 256-aligned offset of the current draw's shadow matrix in the ring
        uint32_t shadow_dynamic_offset = 0;
    };

    // Everything a drawable consumes from the renderer that draws it — and
    // nothing more (the HUD counterpart is HudFrame). The renderer implements
    // this port; the drawables never see the renderer itself.
    class DrawableContext {
    public:
        virtual ~DrawableContext() = default;

        virtual const std::shared_ptr<VulkanDevice> &device() const = 0;
        // thread-confined pool for the drawables' one-shot uploads
        virtual VkCommandPool upload_pool() const = 0;
        virtual DescriptorAllocator &descriptors() = 0;
        virtual VkDescriptorSetLayout set0_plain_layout() const = 0;
        virtual VkDescriptorSetLayout set0_shadow_layout() const = 0;
        virtual VkFormat shadow_depth_format() const = 0;

        // scene-pass attachment setup, for the drawables' lazy pipelines
        virtual VkFormat scene_color_format() const = 0;
        virtual VkFormat scene_depth_format() const = 0;
        virtual VkSampleCountFlagBits scene_samples() const = 0;

        virtual const SceneFrame &scene_frame() const = 0;
    };

}// namespace arenai::view

#endif// ARENAI_VK_DRAWABLE_CONTEXT_H
