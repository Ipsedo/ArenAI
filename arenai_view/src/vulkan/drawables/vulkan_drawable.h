//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VULKAN_DRAWABLE_H
#define ARENAI_VULKAN_DRAWABLE_H

#include <arenai_view/drawable.h>

namespace arenai::view {

    class VulkanRenderer;

    // Internal base of every Vulkan drawable. A drawable belongs to exactly
    // one renderer: add_drawable() attaches it, giving access to the frame
    // being recorded (command buffer, set-0 descriptors, target formats) and
    // to the renderer's thread-confined pools. GPU resources (buffers,
    // pipelines, descriptor sets) are created lazily on first draw, on the
    // renderer's thread — the factory only captures CPU-side data.
    class VulkanDrawable : public AbstractDrawable {
    public:
        void attach(VulkanRenderer *renderer);

    protected:
        VulkanRenderer *renderer_ = nullptr;
    };

}// namespace arenai::view

#endif// ARENAI_VULKAN_DRAWABLE_H
