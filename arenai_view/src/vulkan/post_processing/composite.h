//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_COMPOSITE_H
#define ARENAI_VK_COMPOSITE_H

#include "./effect.h"

namespace arenai::view {

    // Final pass: tonemapping/grading onto the caller's open rendering scope
    // (the swapchain image for the player, an offscreen target in the tests).
    class CompositeEffect final : public VulkanPostEffect {
    public:
        CompositeEffect(
            const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
            int width, int height);

        void render(FrameContext &context) override;
    };

}// namespace arenai::view

#endif// ARENAI_VK_COMPOSITE_H
