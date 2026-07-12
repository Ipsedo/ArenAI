//
// Created by samuel on 13/07/2026.
//

#ifndef ARENAI_SHADOW_DRAWABLE_H
#define ARENAI_SHADOW_DRAWABLE_H

#include <GLES3/gl3.h>
#include <glm/glm.hpp>

#include <arenai_view/drawable.h>

namespace arenai::view {

    // Internal extension of AbstractDrawable for drawables that cast and
    // receive dynamic shadows. The renderer detects it via dynamic_cast, so
    // the public API (include/) stays untouched.
    class GlShadowDrawable : public AbstractDrawable {
    public:
        // depth-only pass, rendered from the light's point of view
        virtual void draw_depth(const glm::mat4 &light_mvp_matrix) = 0;

        // main pass with shadow sampling; shadow_mvp_matrix maps model space
        // to [0, 1] shadow-map coordinates (bias * light_proj * light_view * model)
        virtual void draw_with_shadow(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos, const glm::mat4 &shadow_mvp_matrix,
            GLuint shadow_map_texture) = 0;
    };

}// namespace arenai::view

#endif// ARENAI_SHADOW_DRAWABLE_H
