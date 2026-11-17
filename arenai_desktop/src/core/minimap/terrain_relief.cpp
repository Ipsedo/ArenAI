#include "./terrain_relief.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace arenai::desktop {

    namespace {

        const glm::vec3 GROUND_LOW(18.f, 16.f, 36.f);
        const glm::vec3 GROUND_HIGH(30.f, 27.f, 56.f);
        const glm::vec3 CONTOUR_INK(96.f, 92.f, 150.f);
        const glm::vec3 GRID_INK(242.f, 244.f, 255.f);
        constexpr float CONTOUR_STRENGTH = 0.35f;
        constexpr float GRID_STRENGTH = 0.08f;

    }// namespace

    TerrainRelief::TerrainRelief(utils::ImageGrey height_map, const glm::vec3 scale)
        : height_map_(std::move(height_map)), scale_(scale) {}

    float TerrainRelief::grey_at(const float column, const float row) const {
        const float c = std::clamp(column, 0.f, static_cast<float>(height_map_.width - 1));
        const float r = std::clamp(row, 0.f, static_cast<float>(height_map_.height - 1));
        const int c0 = static_cast<int>(c), r0 = static_cast<int>(r);
        const int c1 = std::min(c0 + 1, height_map_.width - 1);
        const int r1 = std::min(r0 + 1, height_map_.height - 1);
        const float tc = c - static_cast<float>(c0), tr = r - static_cast<float>(r0);

        const auto at = [this](const int col, const int line) {
            return height_map_.pixels[line * height_map_.width + col];
        };
        const float top = at(c0, r0) + (at(c1, r0) - at(c0, r0)) * tc;
        const float bottom = at(c0, r1) + (at(c1, r1) - at(c0, r1)) * tc;
        return top + (bottom - top) * tr;
    }

    gui::MinimapTerrain
    TerrainRelief::render(const float half_extent, const int texels_per_cell) const {
        const int width =
            static_cast<int>(std::round(2.f * half_extent / scale_.x)) * texels_per_cell;
        const int height =
            static_cast<int>(std::round(2.f * half_extent / scale_.z)) * texels_per_cell;
        const float half_columns = static_cast<float>(height_map_.width - 1) * 0.5f;
        const float half_rows = static_cast<float>(height_map_.height - 1) * 0.5f;

        // world x, z of each texel center, relative to the terrain center
        const auto world_x = [&](const int i) {
            return (static_cast<float>(i) + 0.5f) / static_cast<float>(width) * 2.f * half_extent
                   - half_extent;
        };
        const auto world_z = [&](const int j) {
            return (static_cast<float>(j) + 0.5f) / static_cast<float>(height) * 2.f * half_extent
                   - half_extent;
        };

        std::vector<float> greys(static_cast<std::size_t>(width) * height);
        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
                greys[j * width + i] = grey_at(
                    world_x(i) / scale_.x + half_columns, world_z(j) / scale_.z + half_rows);

        const auto [lowest, highest] = std::ranges::minmax(greys);
        const float span = std::max(highest - lowest, 1e-6f);
        const float contour_step = CONTOUR_INTERVAL / scale_.y;

        const auto band = [&](const int i, const int j) {
            return static_cast<int>(std::floor(greys[j * width + i] / contour_step));
        };
        const auto grid_cell = [](const float world) {
            return static_cast<int>(std::floor(world / GRID_SPACING));
        };

        gui::MinimapTerrain terrain{.width = width, .height = height};
        terrain.rgba.resize(static_cast<std::size_t>(width) * height * 4);

        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++) {
                const float altitude = (greys[j * width + i] - lowest) / span;
                glm::vec3 color = GROUND_LOW + (GROUND_HIGH - GROUND_LOW) * altitude;

                const int right = std::min(i + 1, width - 1), down = std::min(j + 1, height - 1);
                if (band(i, j) != band(right, j) || band(i, j) != band(i, down))
                    color += (CONTOUR_INK - color) * CONTOUR_STRENGTH;

                if (grid_cell(world_x(i)) != grid_cell(world_x(right))
                    || grid_cell(world_z(j)) != grid_cell(world_z(down)))
                    color += (GRID_INK - color) * GRID_STRENGTH;

                std::uint8_t *texel = &terrain.rgba[(static_cast<std::size_t>(j) * width + i) * 4];
                for (int c = 0; c < 3; c++)
                    texel[c] = static_cast<std::uint8_t>(std::clamp(color[c], 0.f, 255.f));
                texel[3] = 255;
            }

        return terrain;
    }

}// namespace arenai::desktop
