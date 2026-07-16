//
// Created by samuel on 15/07/2026.
//

#include "./god_rays.h"

#include <algorithm>
#include <cmath>

namespace arenai::view {

    GodRaysEffect::GodRaysEffect(const int width, const int height)
        : AbstractPostProcessingEffect(
            effect_builder("god_rays_fs.glsl")
                .add_uniform("u_depth")
                .add_uniform("u_proj_info")
                .add_uniform("u_sun_uv")
                .add_uniform("u_aspect")
                .build(),
            {{GL_R8, 2}}, width, height) {}

    void GodRaysEffect::render(FrameContext &context) {
        float ray_strength = 0.f;
        glm::vec2 sun_uv(0.5f);
        if (context.sun_dir_view.z < 0.f) {
            const glm::vec4 sun_clip = context.proj_matrix * glm::vec4(context.sun_dir_view, 0.f);
            sun_uv = glm::vec2(sun_clip.x, sun_clip.y) / sun_clip.w * 0.5f + 0.5f;

            const auto outside = [](const float v) { return std::max({0.f, -v, v - 1.f}); };
            const float off_frame = std::max(outside(sun_uv.x), outside(sun_uv.y));

            ray_strength = RAY_STRENGTH * std::min(1.f, -context.sun_dir_view.z / 0.2f)
                           * std::clamp(1.f - off_frame / 0.4f, 0.f, 1.f);
        }

        if (ray_strength > 0.f) {
            program->use();
            program->bind_external_texture("u_depth", context.depth_texture, 0);
            program->uniform_vec4("u_proj_info", context.proj_info);
            program->uniform_vec2("u_sun_uv", sun_uv);
            program->uniform_float(
                "u_aspect", static_cast<float>(context.screen_width)
                                / static_cast<float>(context.screen_height));
            draw_fullscreen(targets[0].fbo, targets[0].width, targets[0].height);
        }

        context.textures["rays"] = targets[0].texture;
        context.scalars["ray_strength"] = ray_strength;
    }

}// namespace arenai::view
