//
// Created by samuel on 03/07/2026.
//

#ifndef ARENAI_DRAWABLE_FACTORY_H
#define ARENAI_DRAWABLE_FACTORY_H
#include <filesystem>
#include <memory>

#include <arenai_utils/file_reader.h>

#include "./drawable.h"

using namespace arenai;

namespace arenai::view {

    class DrawableFactory {
    public:
        virtual ~DrawableFactory() = default;

        virtual std::shared_ptr<Drawable> make_cube_map(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::filesystem::path &pngs_root_path) = 0;

        virtual std::shared_ptr<Drawable> make_specular(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::vector<std::tuple<float, float, float>> &vertices,
            const std::vector<std::tuple<float, float, float>> &normals, glm::vec4 ambient_color,
            glm::vec4 diffuse_color, glm::vec4 specular_color, float shininess) = 0;
    };

}// namespace arenai::view

#endif//ARENAI_DRAWABLE_FACTORY_H
