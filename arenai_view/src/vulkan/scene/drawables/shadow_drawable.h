//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_SHADOW_DRAWABLE_H
#define ARENAI_VK_SHADOW_DRAWABLE_H

#include <glm/glm.hpp>

#include "./vulkan_drawable.h"

namespace arenai::view {

    class VulkanShadowDrawable : public VulkanDrawable {
    public:
        virtual void draw_depth(const glm::mat4 &light_mvp_matrix) = 0;

        virtual void draw_with_shadow(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos, glm::vec4 world_up) = 0;
    };

}// namespace arenai::view

#endif// ARENAI_VK_SHADOW_DRAWABLE_H
