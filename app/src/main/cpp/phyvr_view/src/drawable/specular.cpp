//
// Created by samuel on 18/03/2023.
//

#include <iostream>

#include <phyvr_view/errors.h>
#include <phyvr_view/specular.h>

/*
 * Specular
 */

Specular::Specular(
    const std::shared_ptr<AbstractFileReader> &file_reader,
    const std::vector<std::tuple<float, float, float>> &vertices,
    const std::vector<std::tuple<float, float, float>> &normals, glm::vec4 ambient_color,
    glm::vec4 diffuse_color, glm::vec4 specular_color, float shininess, const std::string &shape_id)
    : ambient_color(ambient_color), diffuse_color(diffuse_color), specular_color(specular_color),
      shininess(shininess), nb_vertices(static_cast<int>(vertices.size())) {

    std::vector<float> vbo_data;
    for (int i = 0; i < vertices.size(); i++) {
        auto [x, y, z] = vertices[i];
        vbo_data.push_back(x);
        vbo_data.push_back(y);
        vbo_data.push_back(z);

        auto [n_x, n_y, n_z] = normals[i];
        vbo_data.push_back(n_x);
        vbo_data.push_back(n_y);
        vbo_data.push_back(n_z);
    }

    program = Program::Builder(file_reader, "shaders/specular_vs.glsl", "shaders/specular_fs.glsl")
                  .add_uniform("u_mvp_matrix")
                  .add_uniform("u_mv_matrix")
                  .add_uniform("u_ambient_color")
                  .add_uniform("u_diffuse_color")
                  .add_uniform("u_specular_color")
                  .add_uniform("u_light_pos")
                  .add_uniform("u_shininess")
                  .add_uniform("u_cam_pos")
                  .add_buffer("vertices_normals_buffer", vbo_data)
                  .add_attribute("a_position")
                  .add_attribute("a_normal")
                  .build();
}

void Specular::draw(
    const glm::mat4 mvp_matrix, const glm::mat4 mv_matrix, const glm::vec3 light_pos_from_camera,
    const glm::vec3 camera_pos) {
    program->use();

    program->attrib("a_position", "vertices_normals_buffer", POSITION_SIZE, STRIDE, 0);
    program->attrib(
        "a_normal", "vertices_normals_buffer", NORMAL_SIZE, STRIDE,
        POSITION_SIZE * BYTES_PER_FLOAT);

    program->uniform_mat4("u_mvp_matrix", mvp_matrix);
    program->uniform_mat4("u_mv_matrix", mv_matrix);

    program->uniform_vec3("u_light_pos", light_pos_from_camera);
    program->uniform_vec3("u_cam_pos", camera_pos);

    program->uniform_vec4("u_ambient_color", ambient_color);
    program->uniform_vec4("u_diffuse_color", diffuse_color);
    program->uniform_vec4("u_specular_color", specular_color);

    program->uniform_float("u_shininess", shininess);

    Program::draw_arrays(GL_TRIANGLES, 0, nb_vertices);

    program->disable_attrib_array();
}

Specular::~Specular() { program = std::nullptr_t(); }
