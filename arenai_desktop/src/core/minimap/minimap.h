#ifndef ARENAI_DESKTOP_CORE_MINIMAP_MINIMAP_H
#define ARENAI_DESKTOP_CORE_MINIMAP_MINIMAP_H

#include <vector>

#include <arenai_core/arena_terrain.h>
#include <arenai_utils/file_reader.h>

#include "../../gui/gui.h"
#include "./minimap_projection.h"
#include "./tank_pose.h"

namespace arenai::desktop {

    // what the HUD's minimap shows, from the tanks' poses
    class Minimap {
    public:
        Minimap(const MinimapProjection &projection, gui::MinimapTerrain terrain);

        const gui::MinimapTerrain &terrain() const;

        // view_forward: the camera's look direction, world space
        gui::MinimapSnapshot snapshot(
            const TankPose &player, glm::vec3 view_forward,
            const std::vector<TankPose> &enemies) const;

        // world units: the shown window's half side
        static constexpr float VIEW_HALF_SIZE = 400.f;
        static constexpr int TEXELS_PER_CELL = 4;
        // frame fraction kept between an out-of-range blip and the border
        static constexpr float EDGE_MARGIN = 0.04f;

    private:
        MinimapProjection projection_;
        gui::MinimapTerrain terrain_;
    };

    // reads the arena's height map and renders its relief
    Minimap
    make_minimap(const core::ArenaTerrain &terrain, utils::AbstractResourceFileReader &reader);

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_MINIMAP_MINIMAP_H
