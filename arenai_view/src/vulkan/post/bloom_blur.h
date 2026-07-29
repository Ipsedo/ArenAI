//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_BLOOM_BLUR_H
#define ARENAI_VK_BLOOM_BLUR_H

#include "./effect.h"

namespace arenai::view {

    class BloomBlurEffect final : public VulkanPostEffect {
    public:
        BloomBlurEffect(
            const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
            int width, int height);

        void render(FrameContext &context) override;
    };

}// namespace arenai::view

#endif// ARENAI_VK_BLOOM_BLUR_H
