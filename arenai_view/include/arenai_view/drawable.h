//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_DRAWABLE_H
#define ARENAI_DRAWABLE_H
#include <filesystem>
#include <memory>
#include <tuple>
#include <vector>

#include <glm/glm.hpp>

#include <arenai_utils/file_reader.h>

namespace arenai::view {

    class AbstractDrawable {
    public:
        virtual void draw(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos) = 0;
        virtual ~AbstractDrawable() = default;
    };

    class AbstractDrawableFactory {
    public:
        virtual ~AbstractDrawableFactory() = default;

        virtual std::unique_ptr<AbstractDrawable> make_cube_map(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::filesystem::path &pngs_root_path) = 0;

        virtual std::unique_ptr<AbstractDrawable> make_diffuse(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::vector<std::tuple<float, float, float>> &vertices, glm::vec4 color) = 0;
    };

}// namespace arenai::view

#endif// ARENAI_DRAWABLE_H
