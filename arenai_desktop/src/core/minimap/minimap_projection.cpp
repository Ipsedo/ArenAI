#include "./minimap_projection.h"

#include <cmath>

namespace arenai::desktop {

    namespace {

        glm::vec2 ground(const glm::vec3 world) { return {world.x, world.z}; }

    }// namespace

    MinimapProjection::MinimapProjection(
        const glm::vec2 terrain_center, const float half_extent, const float view_half_size)
        : terrain_center_(terrain_center), half_extent_(half_extent),
          view_half_size_(view_half_size) {}

    glm::vec2 MinimapProjection::texture_coords(const glm::vec3 world) const {
        return (ground(world) - terrain_center_) / (2.f * half_extent_) + 0.5f;
    }

    std::pair<glm::vec2, glm::vec2> MinimapProjection::view_window(const glm::vec3 player) const {
        const glm::vec2 center = texture_coords(player);
        const glm::vec2 half(view_half_size_ / (2.f * half_extent_));
        return {center - half, center + half};
    }

    glm::vec2
    MinimapProjection::frame_coords(const glm::vec3 player, const glm::vec3 target) const {
        return (ground(target) - ground(player)) / (2.f * view_half_size_) + 0.5f;
    }

    float MinimapProjection::heading(const glm::vec3 forward) {
        constexpr float minimal_length = 1e-6f;
        if (forward.x * forward.x + forward.z * forward.z < minimal_length) return 0.f;
        return std::atan2(forward.x, -forward.z);
    }

}// namespace arenai::desktop
