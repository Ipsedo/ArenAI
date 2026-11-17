#ifndef ARENAI_ARENA_TERRAIN_H
#define ARENAI_ARENA_TERRAIN_H

#include <filesystem>

#include <glm/glm.hpp>

namespace arenai::core {

    struct ArenaTerrain {
        // relative to the asset folder
        std::filesystem::path height_map;
        glm::vec3 position;
        glm::vec3 scale;
    };

}// namespace arenai::core

#endif// ARENAI_ARENA_TERRAIN_H
