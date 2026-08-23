//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VULKAN_DRAWABLE_H
#define ARENAI_VULKAN_DRAWABLE_H

#include <arenai_view/drawable.h>

#include "./drawable_context.h"

namespace arenai::view {

    class VulkanDrawable : public AbstractDrawable {
    public:
        void attach(DrawableContext *context);

    protected:
        DrawableContext *context_ = nullptr;
    };

}// namespace arenai::view

#endif// ARENAI_VULKAN_DRAWABLE_H
