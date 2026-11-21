#ifndef ARENAI_ARENA_TERRAIN_H
#define ARENAI_ARENA_TERRAIN_H

#include <cmath>
#include <filesystem>

#include <glm/glm.hpp>

namespace arenai::core {

    struct ArenaTerrain {
        // relative to the asset folder
        std::filesystem::path height_map;
        glm::vec3 position;
        glm::vec3 scale;
        // world units: half side of the playable square around position
        float half_extent;

        bool contains(const glm::vec3 &world) const {
            return std::abs(world.x - position.x) <= half_extent
                   && std::abs(world.z - position.z) <= half_extent;
        }
    };

}// namespace arenai::core

#endif// ARENAI_ARENA_TERRAIN_H
