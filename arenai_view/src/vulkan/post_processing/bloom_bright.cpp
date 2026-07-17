//
// Created by samuel on 17/07/2026.
//

#include "./bloom_bright.h"

namespace arenai::view {

    BloomBrightEffect::BloomBrightEffect(
        const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
        const int width, const int height)
        : VulkanPostEffect(
              device, descriptors, "bloom_bright_fs.glsl", 1, 0, {{VK_FORMAT_R8G8B8A8_UNORM, 2}},
              width, height) {}

    void BloomBrightEffect::render(FrameContext &context) {
        run_pass(context, 0, {context.scene}, nullptr);

        context.textures["bright"] = target(0);
    }

}// namespace arenai::view
