//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_SPECULAR_H
#define ARENAI_SPECULAR_H

#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include <arenai_utils/file_reader.h>

#include "../constants.h"
#include "../program.h"
#include "./shadow_drawable.h"

namespace arenai::view {

    class Specular final : public GlShadowDrawable {
    public:
        Specular(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::vector<std::tuple<float, float, float>> &vertices,
            const std::vector<std::tuple<float, float, float>> &normals, glm::vec4 ambient_color,
            glm::vec4 diffuse_color, glm::vec4 specular_color, float shininess);

        void draw(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos) override;

        void draw_depth(const glm::mat4 &light_mvp_matrix) override;

        void draw_with_shadow(
            glm::mat4 mvp_matrix, glm::mat4 mv_matrix, glm::vec3 light_pos_from_camera,
            glm::vec3 camera_pos, const glm::mat4 &shadow_mvp_matrix,
            GLuint shadow_map_texture) override;

    private:
        static constexpr int POSITION_SIZE = 3;
        static constexpr int NORMAL_SIZE = 3;
        static constexpr int STRIDE = (POSITION_SIZE + NORMAL_SIZE) * BYTES_PER_FLOAT;

        void bind_specular_pass(
            Program &curr_program, glm::mat4 mvp_matrix, glm::mat4 mv_matrix,
            glm::vec3 light_pos_from_camera, glm::vec3 camera_pos);

        std::shared_ptr<utils::AbstractFileReader> file_reader;
        std::vector<float> vbo_data;

        std::unique_ptr<Program> program;
        // shadow-related programs, built lazily so that renderers without
        // shadows (offscreen agent vision) pay nothing
        std::unique_ptr<Program> depth_program;
        std::unique_ptr<Program> shadow_program;

        glm::vec4 ambient_color;
        glm::vec4 diffuse_color;
        glm::vec4 specular_color;
        float shininess;

        int nb_vertices;
    };

}// namespace arenai::view

#endif// ARENAI_SPECULAR_H
