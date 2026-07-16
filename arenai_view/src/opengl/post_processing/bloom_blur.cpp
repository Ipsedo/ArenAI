//
// Created by samuel on 15/07/2026.
//

#include "./bloom_blur.h"

namespace arenai::view {

    BloomBlurEffect::BloomBlurEffect(const int width, const int height)
        : AbstractPostProcessingEffect(
            effect_builder("bloom_blur_fs.glsl")
                .add_uniform("u_source")
                .add_uniform("u_direction")
                .build(),
            {{GL_RGBA8, 4}, {GL_RGBA8, 4}}, width, height) {}

    void BloomBlurEffect::render(FrameContext &context) {
        program->use();

        program->bind_external_texture("u_source", context.textures.at("bright"), 0);
        program->uniform_vec2("u_direction", glm::vec2(1.f, 0.f));
        draw_fullscreen(targets[0].fbo, targets[0].width, targets[0].height);

        program->bind_external_texture("u_source", targets[0].texture, 0);
        program->uniform_vec2("u_direction", glm::vec2(0.f, 1.f));
        draw_fullscreen(targets[1].fbo, targets[1].width, targets[1].height);

        context.textures["bloom"] = targets[1].texture;
    }

}// namespace arenai::view
