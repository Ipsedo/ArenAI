#ifndef ARENAI_VK_DRAWABLE_CONTEXT_H
#define ARENAI_VK_DRAWABLE_CONTEXT_H

#include <memory>

#include "../../core/descriptors.h"
#include "../../core/device.h"
#include "../../core/vk.h"
#include "../scene_pipelines.h"

namespace arenai::view {

    struct SceneFrame {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        int slot = 0;
        VkDescriptorSet set0_plain = VK_NULL_HANDLE;
        VkDescriptorSet set0_shadow = VK_NULL_HANDLE;
        uint32_t shadow_dynamic_offset = 0;
    };

    class DrawableContext {
    public:
        virtual ~DrawableContext() = default;

        virtual const std::shared_ptr<VulkanDevice> &device() const = 0;

        virtual VkCommandPool upload_pool() const = 0;

        virtual DescriptorAllocator &descriptors() = 0;

        virtual const ScenePipelines &pipelines() const = 0;

        virtual const SceneFrame &scene_frame() const = 0;
    };

}// namespace arenai::view

#endif// ARENAI_VK_DRAWABLE_CONTEXT_H
