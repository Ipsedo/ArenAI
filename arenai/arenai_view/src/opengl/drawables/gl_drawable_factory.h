//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_GL_DRAWABLE_FACTORY_H
#define ARENAI_GL_DRAWABLE_FACTORY_H

#include <arenai_view/drawable_factory.h>

namespace arenai::view {

    class GlDrawableFactory final : public AbstractDrawableFactory {
    public:
        std::unique_ptr<AbstractDrawable> make_cube_map(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::filesystem::path &pngs_root_path) override;

        std::unique_ptr<AbstractDrawable> make_specular(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::vector<std::tuple<float, float, float>> &vertices,
            const std::vector<std::tuple<float, float, float>> &normals, glm::vec4 ambient_color,
            glm::vec4 diffuse_color, glm::vec4 specular_color, float shininess) override;
    };

}// namespace arenai::view

#endif// ARENAI_GL_DRAWABLE_FACTORY_H
