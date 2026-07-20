//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VULKAN_DRAWABLE_H
#define ARENAI_VULKAN_DRAWABLE_H

#include <arenai_view/drawable.h>

#include "./drawable_context.h"

namespace arenai::view {

    // Internal base of every Vulkan drawable. A drawable belongs to exactly
    // one renderer: add_drawable() attaches it to the renderer's
    // DrawableContext port, giving access to the frame being recorded
    // (command buffer, set-0 descriptors, target formats) and to the
    // renderer's thread-confined pools. GPU resources (buffers, pipelines,
    // descriptor sets) are created lazily on first draw, on the renderer's
    // thread — the factory only captures CPU-side data.
    class VulkanDrawable : public AbstractDrawable {
    public:
        void attach(DrawableContext *context);

    protected:
        DrawableContext *context_ = nullptr;
    };

}// namespace arenai::view

#endif// ARENAI_VULKAN_DRAWABLE_H
