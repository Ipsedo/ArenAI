#include "./composite.h"

namespace arenai::view {

    namespace {
        struct CompositePush {
            float ray_strength;
            float frame;
            float gamma_exponent;
        };
    }// namespace

    CompositeEffect::CompositeEffect(
        const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
        const int width, const int height, const float gamma_exponent)
        : VulkanPostEffect(
            device, descriptors, "post_fs.glsl", 4, sizeof(CompositePush), {}, width, height),
          gamma_exponent_(gamma_exponent) {}

    void CompositeEffect::render(FrameContext &context) {
        const CompositePush push{
            .ray_strength = context.scalars.at(PostScalar::god_ray_strength),
            .frame = static_cast<float>(context.frame),
            .gamma_exponent = gamma_exponent_};
        run_inline(
            context,
            {context.scene, context.textures.at(PostTexture::ao),
             context.textures.at(PostTexture::bloom), context.textures.at(PostTexture::god_rays)},
            &push);
    }

}// namespace arenai::view
