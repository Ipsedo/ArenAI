//
// Created by samuel on 15/07/2026.
//

#include "./ssao.h"

namespace arenai::view {

    SsaoEffect::SsaoEffect(const int width, const int height)
        : AbstractPostProcessingEffect(
            effect_builder("ssao_fs.glsl")
                .add_uniform("u_depth")
                .add_uniform("u_proj_info")
                .build(),
            {{GL_R8, 2}}, width, height) {}

    void SsaoEffect::render(FrameContext &context) {
        program->use();
        program->bind_external_texture("u_depth", context.depth_texture, 0);
        program->uniform_vec4("u_proj_info", context.proj_info);
        draw_fullscreen(targets[0].fbo, targets[0].width, targets[0].height);

        context.textures["ao_raw"] = targets[0].texture;
    }

}// namespace arenai::view
