//
// Created by samuel on 17/07/2026.
//

#include "./ao_blur.h"

namespace arenai::view {

    AoBlurEffect::AoBlurEffect(
        const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
        const int width, const int height)
        : VulkanPostEffect(
            device, descriptors, "ao_blur_fs.glsl", 1, 0, {{VK_FORMAT_R8_UNORM, 2}}, width,
            height) {}

    void AoBlurEffect::render(FrameContext &context) {
        run_pass(context, 0, {context.textures.at(PostTexture::ao_raw)}, nullptr);

        context.textures[PostTexture::ao] = target(0);
    }

}// namespace arenai::view
