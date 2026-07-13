//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_DIFFUSE_H
#define ARENAI_DIFFUSE_H

#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include <arenai_utils/file_reader.h>

#include "../constants.h"
#include "../program.h"
#include "./shadow_drawable.h"

namespace arenai::view {

    // Flat-shaded matte drawable: the fragment shader derives per-face normals
    // from screen-space derivatives, so only vertex positions are uploaded.
    class Diffuse final : public GlShadowDrawable {
    public:
        Diffuse(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::vector<std::tuple<float, float, float>> &vertices, glm::vec4 color);

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
        static constexpr int STRIDE = POSITION_SIZE * BYTES_PER_FLOAT;

        void bind_diffuse_pass(
            Program &curr_program, const glm::mat4 &mvp_matrix, const glm::mat4 &mv_matrix,
            glm::vec3 light_pos_from_camera) const;

        std::shared_ptr<utils::AbstractFileReader> file_reader;
        std::vector<float> vbo_data;

        std::unique_ptr<Program> program;
        // shadow-related programs, built lazily so that renderers without
        // shadows (offscreen agent vision) pay nothing
        std::unique_ptr<Program> depth_program;
        std::unique_ptr<Program> shadow_program;

        glm::vec4 color;

        int nb_vertices;
    };

}// namespace arenai::view

#endif// ARENAI_DIFFUSE_H
