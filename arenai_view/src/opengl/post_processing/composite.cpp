//
// Created by samuel on 15/07/2026.
//

#include "./composite.h"

namespace arenai::view {

    CompositeEffect::CompositeEffect(const int width, const int height)
        : AbstractPostProcessingEffect(
            effect_builder("post_fs.glsl")
                .add_uniform("u_scene")
                .add_uniform("u_ao")
                .add_uniform("u_bloom")
                .add_uniform("u_rays")
                .add_uniform("u_ray_strength")
                .add_uniform("u_frame")
                .build(),
            {}, width, height) {}

    void CompositeEffect::render(FrameContext &context) {
        program->use();
        program->bind_external_texture("u_scene", context.scene_texture, 0);
        program->bind_external_texture("u_ao", context.textures.at("ao"), 1);
        program->bind_external_texture("u_bloom", context.textures.at("bloom"), 2);
        program->bind_external_texture("u_rays", context.textures.at("rays"), 3);
        program->uniform_float("u_ray_strength", context.scalars.at("ray_strength"));
        program->uniform_float("u_frame", static_cast<float>(context.frame));
        draw_fullscreen(0, context.screen_width, context.screen_height);
    }

}// namespace arenai::view
