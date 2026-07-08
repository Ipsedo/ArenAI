//
// Created by samuel on 19/03/2023.
//

#ifndef ARENAI_CUBEMAP_H
#define ARENAI_CUBEMAP_H

#include <memory>
#include <string>

#include <arenai_utils/file_reader.h>
#include <arenai_view/drawable.h>

#include "../constants.h"
#include "../program.h"

namespace arenai::view {

    class CubeMap final : public AbstractDrawable {
    public:
        CubeMap(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::filesystem::path &pngs_root_path);

        void draw(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos) override;

    private:
        static constexpr int POSITION_SIZE = 3;
        static constexpr int STRIDE = POSITION_SIZE * BYTES_PER_FLOAT;

        std::unique_ptr<Program> program;

        int nb_vertices;
    };

}// namespace arenai::view

#endif// ARENAI_CUBEMAP_H
