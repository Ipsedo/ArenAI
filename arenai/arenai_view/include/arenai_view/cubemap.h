//
// Created by samuel on 19/03/2023.
//

#ifndef ARENAI_CUBEMAP_H
#define ARENAI_CUBEMAP_H

#include <memory>
#include <string>

#include <arenai_utils/file_reader.h>
#include <arenai_view/constants.h>
#include <arenai_view/drawable.h>
#include <arenai_view/program.h>

class CubeMap final : public Drawable {
private:
    static const int POSITION_SIZE = 3;
    static const int STRIDE = POSITION_SIZE * BYTES_PER_FLOAT;

    std::unique_ptr<Program> program;

    int nb_vertices;

public:
    CubeMap(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &pngs_root_path);

    void draw(
        glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
        glm::vec3 camera_pos) override;
};

#endif// ARENAI_CUBEMAP_H
