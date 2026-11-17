#ifndef ARENAI_DESKTOP_CORE_MINIMAP_TERRAIN_RELIEF_H
#define ARENAI_DESKTOP_CORE_MINIMAP_TERRAIN_RELIEF_H

#include <glm/glm.hpp>

#include <arenai_utils/file_reader.h>

#include "../../gui/gui.h"

namespace arenai::desktop {

    // the height map drawn as dim contour lines over a faint world grid
    class TerrainRelief {
    public:
        // grey levels in [0, 1], scaled like the physics terrain
        TerrainRelief(utils::ImageGrey height_map, glm::vec3 scale);

        // the square of half side half_extent (world units) around the terrain center
        gui::MinimapTerrain render(float half_extent, int texels_per_cell) const;

        // world units
        static constexpr float CONTOUR_INTERVAL = 10.f;
        static constexpr float GRID_SPACING = 100.f;

    private:
        // bilinear, clamped to the image
        float grey_at(float column, float row) const;

        utils::ImageGrey height_map_;
        glm::vec3 scale_;
    };

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_MINIMAP_TERRAIN_RELIEF_H
