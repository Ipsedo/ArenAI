//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_PROGRAM_H
#define ARENAI_PROGRAM_H

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <GLES3/gl3.h>
#include <glm/glm.hpp>

#include <arenai_utils/file_reader.h>

namespace arenai::view {

    class Program {
    public:
        class Builder {
        public:
            Builder(
                const std::shared_ptr<utils::AbstractResourceFileReader> &text_reader,
                const std::string &vertex_shader_name,
                const std::string &fragment_shader_name);

            Builder add_uniform(const std::string &name);

            Builder add_attribute(const std::string &name);

            Builder add_buffer(const std::string &name, const std::vector<float> &data);

            Builder add_cube_texture(
                const std::string &name, const std::filesystem::path &cube_textures_root_path);

            Builder add_texture(const std::string &name, const std::filesystem::path &texture_path);

            std::unique_ptr<Program> build();

        private:
            Builder(
                const std::shared_ptr<utils::AbstractResourceFileReader> &text_reader,
                std::string vertex_shader_name,
                std::string fragment_shader_name, std::vector<std::string> uniforms,
                std::vector<std::string> attributes,
                std::map<std::string, std::vector<float>> buffers,
                std::map<std::string, std::vector<std::filesystem::path>> cube_textures,
                std::map<std::string, std::filesystem::path> textures);

            std::shared_ptr<utils::AbstractResourceFileReader> file_reader;

            std::string vertex_shader_name;
            std::string fragment_shader_name;

            std::vector<std::string> uniforms;
            std::vector<std::string> attributes;

            std::map<std::string, std::vector<float>> buffers;

            std::map<std::string, std::vector<std::filesystem::path>> cube_textures;
            std::map<std::string, std::filesystem::path> textures;
        };

        explicit Program();

    private:
        GLuint program_id;
        GLuint vertex_shader_id;
        GLuint fragment_shader_id;

        std::map<std::string, GLuint> uniform_handles;
        std::map<std::string, GLuint> attribute_handles;

        std::map<std::string, GLuint> buffer_ids;

        static std::map<std::string, GLenum> get_file_to_texture_id();

        std::map<std::string, std::tuple<GLuint, GLuint>> tex_name_to_idx_id;

        template<typename F, class... T>
        void uniform_(F uniform_fun, const std::string &name, T... args);

        void texture_(GLenum texture_target, const std::string &name);

        static void disable_texture_(GLenum texture_target);

    public:
        void use() const;

        void uniform_mat4(const std::string &name, glm::mat4 mat4);

        void uniform_vec4(const std::string &name, glm::vec4 vec4);

        void uniform_vec3(const std::string &name, glm::vec3 vec3);

        void uniform_vec2(const std::string &name, glm::vec2 vec2);

        void uniform_float(const std::string &name, float f);

        void attrib(
            const std::string &name, const std::string &buffer_name, int data_size, int stride,
            int offset);

        void disable_attrib_array();

        void cube_texture(const std::string &cube_texture_name);

        void texture(const std::string &texture_name);

        void bind_external_texture(const std::string &name, GLuint texture_id, int texture_unit);

        static void disable_cube_texture();

        static void disable_texture();

        static void draw_arrays(GLenum type, int from, int nb_vertices);

        ~Program();
    };

}// namespace arenai::view

#endif// ARENAI_PROGRAM_H
