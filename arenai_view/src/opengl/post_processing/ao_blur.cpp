//
// Created by samuel on 15/07/2026.
//

#include "./ao_blur.h"

namespace arenai::view {

    AoBlurEffect::AoBlurEffect(const int width, const int height)
        : AbstractPostProcessingEffect(
            effect_builder("ao_blur_fs.glsl").add_uniform("u_ao").build(), {{GL_R8, 2}}, width,
            height) {}

    void AoBlurEffect::render(FrameContext &context) {
        program->use();
        program->bind_external_texture("u_ao", context.textures.at("ao_raw"), 0);
        draw_fullscreen(targets[0].fbo, targets[0].width, targets[0].height);

        context.textures["ao"] = targets[0].texture;
    }

}// namespace arenai::view
