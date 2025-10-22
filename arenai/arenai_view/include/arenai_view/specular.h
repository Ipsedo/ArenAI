//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_SPECULAR_H
#define ARENAI_SPECULAR_H

#include <memory>

#include <glm/glm.hpp>

#include <arenai_utils/file_reader.h>
#include <arenai_view/constants.h>
#include <arenai_view/drawable.h>
#include <arenai_view/program.h>

class Specular final : public Drawable {
private:
    static const int POSITION_SIZE = 3;
    static const int NORMAL_SIZE = 3;
    static const int STRIDE = (POSITION_SIZE + NORMAL_SIZE) * BYTES_PER_FLOAT;

    std::unique_ptr<Program> program;

    glm::vec4 ambient_color;
    glm::vec4 diffuse_color;
    glm::vec4 specular_color;
    float shininess;

    int nb_vertices;

public:
    Specular(
        const std::shared_ptr<AbstractFileReader> &file_reader,
        const std::vector<std::tuple<float, float, float>> &vertices,
        const std::vector<std::tuple<float, float, float>> &normals, glm::vec4 ambient_color,
        glm::vec4 diffuse_color, glm::vec4 specular_color, float shininess,
        const std::string &shape_id);

    void draw(
        glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
        glm::vec3 camera_pos) override;
};

#endif// ARENAI_SPECULAR_H
