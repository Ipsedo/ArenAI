//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_CUBEMAP_H
#define PHYVR_CUBEMAP_H

#include <memory>
#include <string>

#include <phyvr_utils/file_reader.h>
#include <phyvr_view/constants.h>
#include <phyvr_view/drawable.h>
#include <phyvr_view/program.h>

class CubeMap : public Drawable {
private:
    static const int POSITION_SIZE = 3;
    static const int STRIDE = POSITION_SIZE * BYTES_PER_FLOAT;

    std::shared_ptr<Program> program;

    int nb_vertices;

public:
    CubeMap(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &pngs_root_path);

    void draw(
        glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
        glm::vec3 camera_pos) override;
};

#endif// PHYVR_CUBEMAP_H
