//
// Created by samuel on 18/03/2023.
//

#include "./specular.h"

#include <iostream>

#include "../errors.h"

using namespace arenai;

namespace arenai::view {

    /*
     * Specular
     */

    Specular::Specular(
        const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        const std::vector<std::tuple<float, float, float>> &vertices,
        const std::vector<std::tuple<float, float, float>> &normals, glm::vec4 ambient_color,
        glm::vec4 diffuse_color, glm::vec4 specular_color, float shininess)
        : file_reader(file_reader), ambient_color(ambient_color), diffuse_color(diffuse_color),
          specular_color(specular_color), shininess(shininess),
          nb_vertices(static_cast<int>(vertices.size())) {

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

        program = Program::Builder(
                      file_reader, std::filesystem::path("shaders") / "specular_vs.glsl",
                      std::filesystem::path("shaders") / "specular_fs.glsl")
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

    void Specular::bind_specular_pass(
        Program &curr_program, const glm::mat4 mvp_matrix, const glm::mat4 mv_matrix,
        const glm::vec3 light_pos_from_camera, const glm::vec3 camera_pos) {
        curr_program.use();

        curr_program.attrib("a_position", "vertices_normals_buffer", POSITION_SIZE, STRIDE, 0);
        curr_program.attrib(
            "a_normal", "vertices_normals_buffer", NORMAL_SIZE, STRIDE,
            POSITION_SIZE * BYTES_PER_FLOAT);

        curr_program.uniform_mat4("u_mvp_matrix", mvp_matrix);
        curr_program.uniform_mat4("u_mv_matrix", mv_matrix);

        curr_program.uniform_vec3("u_light_pos", light_pos_from_camera);
        curr_program.uniform_vec3("u_cam_pos", camera_pos);

        curr_program.uniform_vec4("u_ambient_color", ambient_color);
        curr_program.uniform_vec4("u_diffuse_color", diffuse_color);
        curr_program.uniform_vec4("u_specular_color", specular_color);

        curr_program.uniform_float("u_shininess", shininess);
    }

    void Specular::draw(
        const glm::mat4 mvp_matrix, const glm::mat4 mv_matrix,
        const glm::vec3 light_pos_from_camera, const glm::vec3 camera_pos) {
        bind_specular_pass(*program, mvp_matrix, mv_matrix, light_pos_from_camera, camera_pos);

        Program::draw_arrays(GL_TRIANGLES, 0, nb_vertices);

        program->disable_attrib_array();
    }

    void Specular::draw_depth(const glm::mat4 &light_mvp_matrix) {
        if (!depth_program)
            depth_program =
                Program::Builder(
                    file_reader, std::filesystem::path("shaders") / "shadow_depth_vs.glsl",
                    std::filesystem::path("shaders") / "shadow_depth_fs.glsl")
                    .add_uniform("u_light_mvp_matrix")
                    .add_buffer("vertices_normals_buffer", vbo_data)
                    .add_attribute("a_position")
                    .build();

        depth_program->use();

        depth_program->attrib("a_position", "vertices_normals_buffer", POSITION_SIZE, STRIDE, 0);
        depth_program->uniform_mat4("u_light_mvp_matrix", light_mvp_matrix);

        Program::draw_arrays(GL_TRIANGLES, 0, nb_vertices);

        depth_program->disable_attrib_array();
    }

    void Specular::draw_with_shadow(
        const glm::mat4 mvp_matrix, const glm::mat4 mv_matrix,
        const glm::vec3 light_pos_from_camera, const glm::vec3 camera_pos,
        const glm::mat4 &shadow_mvp_matrix, const GLuint shadow_map_texture) {
        if (!shadow_program)
            shadow_program =
                Program::Builder(
                    file_reader, std::filesystem::path("shaders") / "specular_shadow_vs.glsl",
                    std::filesystem::path("shaders") / "specular_shadow_fs.glsl")
                    .add_uniform("u_mvp_matrix")
                    .add_uniform("u_mv_matrix")
                    .add_uniform("u_shadow_mvp_matrix")
                    .add_uniform("u_ambient_color")
                    .add_uniform("u_diffuse_color")
                    .add_uniform("u_specular_color")
                    .add_uniform("u_light_pos")
                    .add_uniform("u_shininess")
                    .add_uniform("u_cam_pos")
                    .add_uniform("u_shadow_map")
                    .add_buffer("vertices_normals_buffer", vbo_data)
                    .add_attribute("a_position")
                    .add_attribute("a_normal")
                    .build();

        bind_specular_pass(
            *shadow_program, mvp_matrix, mv_matrix, light_pos_from_camera, camera_pos);

        shadow_program->uniform_mat4("u_shadow_mvp_matrix", shadow_mvp_matrix);
        shadow_program->bind_external_texture("u_shadow_map", shadow_map_texture, 0);

        Program::draw_arrays(GL_TRIANGLES, 0, nb_vertices);

        shadow_program->disable_attrib_array();
        Program::disable_texture();
    }

}// namespace arenai::view
