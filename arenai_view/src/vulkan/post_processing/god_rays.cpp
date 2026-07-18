//
// Created by samuel on 17/07/2026.
//

#include "./god_rays.h"

#include <algorithm>
#include <cmath>

namespace arenai::view {

    namespace {
        struct GodRaysPush {
            glm::vec4 proj_info;
            glm::vec2 sun_uv;
            float aspect;
        };
        static_assert(sizeof(GodRaysPush) == 28);
    }// namespace

    GodRaysEffect::GodRaysEffect(
        const std::shared_ptr<VulkanDevice> &device, DescriptorAllocator *descriptors,
        const int width, const int height)
        : VulkanPostEffect(
            device, descriptors, "god_rays_fs.glsl", 1, sizeof(GodRaysPush),
            {{VK_FORMAT_R8_UNORM, 2}}, width, height) {}

    void GodRaysEffect::render(FrameContext &context) {
        float ray_strength = 0.f;
        glm::vec2 sun_uv(0.5f);
        if (context.sun_dir_view.z < 0.f) {
            const glm::vec4 sun_clip = context.proj_matrix * glm::vec4(context.sun_dir_view, 0.f);
            // top-left-origin uv space (the images are stored top-down): the
            // y term is flipped compared to the GL formula
            sun_uv = glm::vec2(
                sun_clip.x / sun_clip.w * 0.5f + 0.5f, 0.5f - sun_clip.y / sun_clip.w * 0.5f);

            const auto outside = [](const float v) { return std::max({0.f, -v, v - 1.f}); };
            const float off_frame = std::max(outside(sun_uv.x), outside(sun_uv.y));

            ray_strength = RAY_STRENGTH * std::min(1.f, -context.sun_dir_view.z / 0.2f)
                           * std::clamp(1.f - off_frame / 0.4f, 0.f, 1.f);
        }

        if (ray_strength > 0.f) {
            const GodRaysPush push{
                context.proj_info, sun_uv,
                static_cast<float>(context.screen_width)
                    / static_cast<float>(context.screen_height)};
            run_pass(context, 0, {context.depth}, &push);
        } else {
            // pass skipped: the composite still samples the target (weighted
            // by ray_strength = 0), its layout must stay valid
            ensure_target_readable(context, 0);
        }

        context.textures["rays"] = target(0);
        context.scalars["ray_strength"] = ray_strength;
    }

}// namespace arenai::view
