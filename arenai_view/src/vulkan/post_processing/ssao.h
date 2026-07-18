//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_SSAO_H
#define ARENAI_VK_SSAO_H

#include "./effect.h"

namespace arenai::view {

    class SsaoEffect final : public VulkanPostEffect {
    public:
        SsaoEffect(
            const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
            int width, int height);

        void render(FrameContext &context) override;
    };

}// namespace arenai::view

#endif// ARENAI_VK_SSAO_H
