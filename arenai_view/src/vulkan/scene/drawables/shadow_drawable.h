//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_SHADOW_DRAWABLE_H
#define ARENAI_VK_SHADOW_DRAWABLE_H

#include <glm/glm.hpp>

#include "./vulkan_drawable.h"

namespace arenai::view {

    // Internal extension of AbstractDrawable for drawables that cast and
    // receive dynamic shadows. The renderer resolves this capability once,
    // when the drawable is added, so the public API (include/) stays
    // untouched. The shadow-map sampler and the per-draw shadow matrix
    // travel through the renderer's set 0 (written by the renderer before
    // each call), not through parameters.
    class VulkanShadowDrawable : public VulkanDrawable {
    public:
        // depth-only pass, rendered from the light's point of view
        virtual void draw_depth(const glm::mat4 &light_mvp_matrix) = 0;

        // main pass with shadow sampling; world_up carries the world up axis
        // in view space (xyz) and the camera world height (w)
        virtual void draw_with_shadow(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos, glm::vec4 world_up) = 0;
    };

}// namespace arenai::view

#endif// ARENAI_VK_SHADOW_DRAWABLE_H
