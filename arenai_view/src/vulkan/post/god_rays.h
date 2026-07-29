//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_GOD_RAYS_H
#define ARENAI_VK_GOD_RAYS_H

#include "./effect.h"

namespace arenai::view {

    class GodRaysEffect final : public VulkanPostEffect {
    public:
        GodRaysEffect(
            const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
            int width, int height);

        void render(FrameContext &context) override;

    private:
        static constexpr float RAY_STRENGTH = 0.4f;
    };

}// namespace arenai::view

#endif// ARENAI_VK_GOD_RAYS_H
