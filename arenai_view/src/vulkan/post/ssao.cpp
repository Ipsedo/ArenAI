//
// Created by samuel on 17/07/2026.
//

#include "./ssao.h"

namespace arenai::view {

    SsaoEffect::SsaoEffect(
        const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
        const int width, const int height)
        : VulkanPostEffect(
            device, descriptors, "ssao_fs.glsl", 1, sizeof(glm::vec4), {{VK_FORMAT_R8_UNORM, 2}},
            width, height) {}

    void SsaoEffect::render(FrameContext &context) {
        run_pass(context, 0, {context.depth}, &context.proj_info);

        context.textures[PostTexture::ao_raw] = target(0);
    }

}// namespace arenai::view
