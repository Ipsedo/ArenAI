#include "./minimap.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "./terrain_relief.h"

namespace arenai::desktop {

    namespace {

        bool inside_frame(const glm::vec2 position, const float margin) {
            return position.x >= margin && position.x <= 1.f - margin && position.y >= margin
                   && position.y <= 1.f - margin;
        }

        // along the ray from the frame center, onto the margin's square
        glm::vec2 onto_edge(const glm::vec2 position, const float margin) {
            const glm::vec2 offset = position - 0.5f;
            const float reach = 0.5f - margin;
            float scale = std::numeric_limits<float>::infinity();
            for (int axis = 0; axis < 2; axis++)
                if (std::abs(offset[axis]) > 0.f)
                    scale = std::min(scale, reach / std::abs(offset[axis]));
            return glm::vec2(0.5f) + offset * scale;
        }

    }// namespace

    Minimap::Minimap(const MinimapProjection &projection, gui::MinimapTerrain terrain)
        : projection_(projection), terrain_(std::move(terrain)) {}

    const gui::MinimapTerrain &Minimap::terrain() const { return terrain_; }

    gui::MinimapSnapshot
    Minimap::snapshot(const TankPose &player, const std::vector<TankPose> &enemies) const {
        gui::MinimapSnapshot snapshot;
        std::tie(snapshot.view_min, snapshot.view_max) = projection_.view_window(player.position);
        snapshot.player_heading = MinimapProjection::heading(player.forward);
        snapshot.player_coordinates =
            glm::ivec2(glm::round(glm::vec2(player.position.x, player.position.z)));

        snapshot.blips.reserve(enemies.size());
        for (const auto &enemy: enemies) {
            const glm::vec2 position = projection_.frame_coords(player.position, enemy.position);
            if (inside_frame(position, EDGE_MARGIN))
                snapshot.blips.push_back(
                    {position, enemy.alive ? gui::BlipKind::Enemy : gui::BlipKind::DeadEnemy});
            else if (enemy.alive)
                snapshot.blips.push_back(
                    {onto_edge(position, EDGE_MARGIN), gui::BlipKind::EnemyOutOfRange});
        }
        return snapshot;
    }

    Minimap
    make_minimap(const core::ArenaTerrain &terrain, utils::AbstractResourceFileReader &reader) {
        const TerrainRelief relief(
            utils::AbstractResourceFileReader::to_img_grey(reader.read_png(terrain.height_map)),
            terrain.scale);
        return {
            MinimapProjection(
                glm::vec2(terrain.position.x, terrain.position.z), Minimap::HALF_EXTENT,
                Minimap::VIEW_HALF_SIZE),
            relief.render(Minimap::HALF_EXTENT, Minimap::TEXELS_PER_CELL)};
    }

}// namespace arenai::desktop
