//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_COMPOSITE_H
#define ARENAI_VK_COMPOSITE_H

#include "./effect.h"

namespace arenai::view {

    class CompositeEffect final : public VulkanPostEffect {
    public:
        CompositeEffect(
            const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
            int width, int height);

        void render(FrameContext &context) override;
    };

}// namespace arenai::view

#endif// ARENAI_VK_COMPOSITE_H
