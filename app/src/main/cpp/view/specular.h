//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_SPECULAR_H
#define PHYVR_SPECULAR_H

#include "drawable.h"
#include "program.h"
#include "constants.h"

#include <glm/glm.hpp>

class Specular : public Drawable {
private:
    static const int POSITION_SIZE = 3;
    static const int NORMAL_SIZE = 3;
    static const int STRIDE = (POSITION_SIZE + NORMAL_SIZE) * BYTES_PER_FLOAT;

    Program program;

    glm::vec4 ambient_color;
    glm::vec4 diffuse_color;
    glm::vec4 specular_color;
    float shininess;

    int nb_vertices;


public:
    Specular(
            AAssetManager *mgr,
            const std::vector<std::tuple<float, float, float>> &vertices,
            const std::vector<std::tuple<float, float, float>> &normals,
            glm::vec4 ambient_color,
            glm::vec4 diffuse_color,
            glm::vec4 specular_color, float shininess);

    void
    draw(glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
         glm::vec3 camera_pos) override;

    ~Specular();
};

#endif //PHYVR_SPECULAR_H
