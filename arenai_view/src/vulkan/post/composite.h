#ifndef ARENAI_VK_COMPOSITE_H
#define ARENAI_VK_COMPOSITE_H

#include "./effect.h"

namespace arenai::view {

    class CompositeEffect final : public VulkanPostEffect {
    public:
        CompositeEffect(
            const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
            int width, int height, float gamma_exponent);

        void render(FrameContext &context) override;

    private:
        float gamma_exponent_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_COMPOSITE_H
