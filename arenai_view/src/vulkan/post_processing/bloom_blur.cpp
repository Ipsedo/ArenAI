//
// Created by samuel on 17/07/2026.
//

#include "./bloom_blur.h"

namespace arenai::view {

    BloomBlurEffect::BloomBlurEffect(
        const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
        const int width, const int height)
        : VulkanPostEffect(
            device, descriptors, "bloom_blur_fs.glsl", 1, sizeof(glm::vec2),
            {{VK_FORMAT_R8G8B8A8_UNORM, 4}, {VK_FORMAT_R8G8B8A8_UNORM, 4}}, width, height) {}

    void BloomBlurEffect::render(FrameContext &context) {
        // separable gaussian: horizontal into target 0, vertical into target 1
        const glm::vec2 horizontal(1.f, 0.f);
        run_pass(context, 0, {context.textures.at("bright")}, &horizontal);

        const glm::vec2 vertical(0.f, 1.f);
        run_pass(context, 1, {target(0)}, &vertical);

        context.textures["bloom"] = target(1);
    }

}// namespace arenai::view
