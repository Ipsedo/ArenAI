//
// Created by samuel on 18/03/2023.
//

#include "./diffuse.h"

#include <iostream>

#include "../errors.h"

using namespace arenai;

namespace arenai::view {

    /*
     * Diffuse
     */

    // matches the horizon of the "cubemap/1" sky, keeps distant geometry
    // blending into it
    constexpr glm::vec3 FOG_COLOR(0.53f, 0.57f, 0.65f);

    Diffuse::Diffuse(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::vector<std::tuple<float, float, float>> &vertices, const glm::vec4 color)
        : file_reader(file_reader), color(color), nb_vertices(static_cast<int>(vertices.size())) {

        for (const auto &[x, y, z]: vertices) {
            vbo_data.push_back(x);
            vbo_data.push_back(y);
            vbo_data.push_back(z);
        }

        program = Program::Builder(
                      file_reader, "diffuse_vs.glsl",
                      "diffuse_fs.glsl")
                      .add_uniform("u_mvp_matrix")
                      .add_uniform("u_mv_matrix")
                      .add_uniform("u_color")
                      .add_uniform("u_fog_color")
                      .add_uniform("u_light_pos")
                      .add_buffer("vertices_buffer", vbo_data)
                      .add_attribute("a_position")
                      .build();
    }

    void Diffuse::bind_diffuse_pass(
        Program &curr_program, const glm::mat4 &mvp_matrix, const glm::mat4 &mv_matrix,
        const glm::vec3 light_pos_from_camera) const {
        curr_program.use();

        curr_program.attrib("a_position", "vertices_buffer", POSITION_SIZE, STRIDE, 0);

        curr_program.uniform_mat4("u_mvp_matrix", mvp_matrix);
        curr_program.uniform_mat4("u_mv_matrix", mv_matrix);

        curr_program.uniform_vec3("u_light_pos", light_pos_from_camera);

        curr_program.uniform_vec4("u_color", color);
        curr_program.uniform_vec3("u_fog_color", FOG_COLOR);
    }

    void Diffuse::draw(
        const glm::mat4 mvp_matrix, const glm::mat4 mv_matrix,
        const glm::vec3 light_pos_from_camera, const glm::vec3 camera_pos) {
        bind_diffuse_pass(*program, mvp_matrix, mv_matrix, light_pos_from_camera);

        Program::draw_arrays(GL_TRIANGLES, 0, nb_vertices);

        program->disable_attrib_array();
    }

    void Diffuse::draw_depth(const glm::mat4 &light_mvp_matrix) {
        if (!depth_program)
            depth_program = Program::Builder(
                                file_reader, "shadow_depth_vs.glsl",
                                "shadow_depth_fs.glsl")
                                .add_uniform("u_light_mvp_matrix")
                                .add_buffer("vertices_buffer", vbo_data)
                                .add_attribute("a_position")
                                .build();

        depth_program->use();

        depth_program->attrib("a_position", "vertices_buffer", POSITION_SIZE, STRIDE, 0);
        depth_program->uniform_mat4("u_light_mvp_matrix", light_mvp_matrix);

        Program::draw_arrays(GL_TRIANGLES, 0, nb_vertices);

        depth_program->disable_attrib_array();
    }

    void Diffuse::draw_with_shadow(
        const glm::mat4 mvp_matrix, const glm::mat4 mv_matrix,
        const glm::vec3 light_pos_from_camera, const glm::vec3 camera_pos, const glm::vec4 world_up,
        const glm::mat4 &shadow_mvp_matrix, const GLuint shadow_map_texture) {
        if (!shadow_program)
            shadow_program = Program::Builder(
                                 file_reader, "diffuse_shadow_vs.glsl",
                                 "diffuse_shadow_fs.glsl")
                                 .add_uniform("u_mvp_matrix")
                                 .add_uniform("u_mv_matrix")
                                 .add_uniform("u_shadow_mvp_matrix")
                                 .add_uniform("u_color")
                                 .add_uniform("u_fog_color")
                                 .add_uniform("u_light_pos")
                                 .add_uniform("u_world_up")
                                 .add_uniform("u_shadow_map")
                                 .add_buffer("vertices_buffer", vbo_data)
                                 .add_attribute("a_position")
                                 .build();

        bind_diffuse_pass(*shadow_program, mvp_matrix, mv_matrix, light_pos_from_camera);

        shadow_program->uniform_vec4("u_world_up", world_up);
        shadow_program->uniform_mat4("u_shadow_mvp_matrix", shadow_mvp_matrix);
        shadow_program->bind_external_texture("u_shadow_map", shadow_map_texture, 0);

        Program::draw_arrays(GL_TRIANGLES, 0, nb_vertices);

        shadow_program->disable_attrib_array();
        Program::disable_texture();
    }

}// namespace arenai::view
