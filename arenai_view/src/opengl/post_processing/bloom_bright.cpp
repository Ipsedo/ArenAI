//
// Created by samuel on 15/07/2026.
//

#include "./bloom_bright.h"

namespace arenai::view {

    BloomBrightEffect::BloomBrightEffect(const int width, const int height)
        : AbstractPostProcessingEffect(
            effect_builder("bloom_bright_fs.glsl").add_uniform("u_scene").build(), {{GL_RGBA8, 2}},
            width, height) {}

    void BloomBrightEffect::render(FrameContext &context) {
        program->use();
        program->bind_external_texture("u_scene", context.scene_texture, 0);
        draw_fullscreen(targets[0].fbo, targets[0].width, targets[0].height);

        context.textures["bright"] = targets[0].texture;
    }

}// namespace arenai::view
