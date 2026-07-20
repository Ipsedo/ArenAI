//
// Created by samuel on 17/07/2026.
//

#include "./composite.h"

namespace arenai::view {

    namespace {
        struct CompositePush {
            float ray_strength;
            float frame;
        };
    }// namespace

    CompositeEffect::CompositeEffect(
        const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
        const int width, const int height)
        : VulkanPostEffect(
            device, descriptors, "post_fs.glsl", 4, sizeof(CompositePush), {}, width, height) {}

    void CompositeEffect::render(FrameContext &context) {
        const CompositePush push{
            context.scalars.at(PostScalar::god_ray_strength), static_cast<float>(context.frame)};
        run_inline(
            context,
            {context.scene, context.textures.at(PostTexture::ao),
             context.textures.at(PostTexture::bloom), context.textures.at(PostTexture::god_rays)},
            &push);
    }

}// namespace arenai::view
