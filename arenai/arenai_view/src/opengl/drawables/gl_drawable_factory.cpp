//
// Created by samuel on 08/07/2026.
//

#include "./gl_drawable_factory.h"

#include "./cube_map.h"
#include "./specular.h"

namespace arenai::view {

    std::unique_ptr<AbstractDrawable> GlDrawableFactory::make_cube_map(
        const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        const std::filesystem::path &pngs_root_path) {
        return std::make_unique<CubeMap>(file_reader, pngs_root_path);
    }

    std::unique_ptr<AbstractDrawable> GlDrawableFactory::make_specular(
        const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        const std::vector<std::tuple<float, float, float>> &vertices,
        const std::vector<std::tuple<float, float, float>> &normals, glm::vec4 ambient_color,
        glm::vec4 diffuse_color, glm::vec4 specular_color, float shininess) {
        return std::make_unique<Specular>(
            file_reader, vertices, normals, ambient_color, diffuse_color, specular_color,
            shininess);
    }

}// namespace arenai::view
