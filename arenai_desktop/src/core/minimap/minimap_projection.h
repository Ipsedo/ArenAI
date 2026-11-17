#ifndef ARENAI_DESKTOP_CORE_MINIMAP_MINIMAP_PROJECTION_H
#define ARENAI_DESKTOP_CORE_MINIMAP_MINIMAP_PROJECTION_H

#include <utility>

#include <glm/glm.hpp>

namespace arenai::desktop {

    // world (x, z) to the minimap's texture and frame coordinates, north (-z) up
    class MinimapProjection {
    public:
        // world units: the terrain square's half side and the shown window's
        MinimapProjection(glm::vec2 terrain_center, float half_extent, float view_half_size);

        glm::vec2 texture_coords(glm::vec3 world) const;

        // (min, max) in texture coordinates, centered on the player
        std::pair<glm::vec2, glm::vec2> view_window(glm::vec3 player) const;

        // [0, 1]² inside the window, the player at (0.5, 0.5)
        glm::vec2 frame_coords(glm::vec3 player, glm::vec3 target) const;

        // radians, clockwise from north
        static float heading(glm::vec3 forward);

    private:
        glm::vec2 terrain_center_;
        float half_extent_;
        float view_half_size_;
    };

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_MINIMAP_MINIMAP_PROJECTION_H
