//
// Created by samuel on 08/07/2026.
//

#include "./gl_drawable_factory.h"

#include "./cube_map.h"
#include "./diffuse.h"

namespace arenai::view {

    std::unique_ptr<AbstractDrawable> GlDrawableFactory::make_cube_map(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::filesystem::path &pngs_root_path) {
        return std::make_unique<CubeMap>(file_reader, pngs_root_path);
    }

    std::unique_ptr<AbstractDrawable> GlDrawableFactory::make_diffuse(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::vector<std::tuple<float, float, float>> &vertices, const glm::vec4 color) {
        return std::make_unique<Diffuse>(file_reader, vertices, color);
    }

}// namespace arenai::view
