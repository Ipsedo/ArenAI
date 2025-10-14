//
// Created by samuel on 18/03/2023.
//

#include <filesystem>
#include <ranges>
#include <utility>

#include <glm/gtc/type_ptr.hpp>

#include <phyvr_utils/file_reader.h>
#include <phyvr_utils/logging.h>
#include <phyvr_utils/string_utils.h>
#include <phyvr_view/constants.h>
#include <phyvr_view/errors.h>
#include <phyvr_view/program.h>

#include "./shader.h"

/*
 * Buffer builder
 */

Program::Builder::Builder(
    const std::shared_ptr<AbstractFileReader> &text_reader, std::string vertex_shader_path,
    std::string fragment_shader_path)
    : file_reader(text_reader), vertex_shader_path(std::move(vertex_shader_path)),
      fragment_shader_path(std::move(fragment_shader_path)) {}

Program::Builder::Builder(
    const std::shared_ptr<AbstractFileReader> &text_reader, std::string vertex_shader_path,
    std::string fragment_shader_path, std::vector<std::string> uniforms,
    std::vector<std::string> attributes, std::map<std::string, std::vector<float>> buffers,
    std::map<std::string, std::vector<std::filesystem::path>> cube_textures,
    std::map<std::string, std::filesystem::path> textures)
    : file_reader(text_reader), vertex_shader_path(std::move(vertex_shader_path)),
      fragment_shader_path(std::move(fragment_shader_path)), uniforms(std::move(uniforms)),
      attributes(std::move(attributes)), buffers(std::move(buffers)),
      cube_textures(std::move(cube_textures)), textures(std::move(textures)) {}

Program::Builder Program::Builder::add_uniform(const std::string &name) {
    uniforms.push_back(name);
    return {file_reader,          vertex_shader_path,
            fragment_shader_path, uniforms,
            attributes,           buffers,
            cube_textures,        textures};
}

Program::Builder Program::Builder::add_attribute(const std::string &name) {
    attributes.push_back(name);
    return {file_reader,          vertex_shader_path,
            fragment_shader_path, uniforms,
            attributes,           buffers,
            cube_textures,        textures};
}

Program::Builder
Program::Builder::add_buffer(const std::string &name, const std::vector<float> &data) {
    buffers.insert({name, data});
    return {file_reader,          vertex_shader_path,
            fragment_shader_path, uniforms,
            attributes,           buffers,
            cube_textures,        textures};
}

Program::Builder Program::Builder::add_cube_texture(
    const std::string &name, const std::string &cube_textures_root_path) {

    const std::filesystem::path root_path(cube_textures_root_path);

    std::vector<std::filesystem::path> full_paths;
    for (const auto &[png_file, _]: Program::get_file_to_texture_id())
        full_paths.push_back(root_path / png_file);

    cube_textures.insert({name, full_paths});

    return {file_reader,          vertex_shader_path,
            fragment_shader_path, uniforms,
            attributes,           buffers,
            cube_textures,        textures};
}

Program::Builder
Program::Builder::add_texture(const std::string &name, const std::string &texture_path) {
    textures.insert({name, std::filesystem::path(texture_path)});
    return {file_reader,          vertex_shader_path,
            fragment_shader_path, uniforms,
            attributes,           buffers,
            cube_textures,        textures};
}

std::unique_ptr<Program> Program::Builder::build() {
    auto program = std::make_unique<Program>();

    program->program_id = glCreateProgram();

    program->vertex_shader_id = load_shader(file_reader, GL_VERTEX_SHADER, vertex_shader_path);
    program->fragment_shader_id =
        load_shader(file_reader, GL_FRAGMENT_SHADER, fragment_shader_path);

    glAttachShader(program->program_id, program->vertex_shader_id);
    glAttachShader(program->program_id, program->fragment_shader_id);

    glLinkProgram(program->program_id);

    // buffers
    for (const auto &[name, data]: buffers) {
        program->buffer_ids.insert({name, 0});

        glGenBuffers(1, &program->buffer_ids[name]);

        glBindBuffer(GL_ARRAY_BUFFER, program->buffer_ids[name]);
        glBufferData(
            GL_ARRAY_BUFFER, static_cast<int>(data.size()) * BYTES_PER_FLOAT, &data[0],
            GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // uniforms
    for (const auto &name: uniforms)
        program->uniform_handles.insert(
            {name, glGetUniformLocation(program->program_id, name.c_str())});

    // attributes
    for (const auto &name: attributes)
        program->attribute_handles.insert(
            {name, glGetAttribLocation(program->program_id, name.c_str())});
    // textures
    GLenum curr_texture = GL_TEXTURE0;

    // cube textures
    std::map<std::string, GLenum> file_to_tex_id = Program::get_file_to_texture_id();
    for (const auto &[name, files_path]: cube_textures) {
        GLuint tex_id;

        program->uniform_handles.insert(
            {name, glGetUniformLocation(program->program_id, name.c_str())});

        glGenTextures(1, &tex_id);
        glActiveTexture(curr_texture);
        glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id);

        program->tex_name_to_idx_id.insert({name, {curr_texture, tex_id}});

        for (const auto &file: files_path) {
            ImageChannels img = file_reader->read_png(file);

            if (img.channels != 4 && img.channels != 3)
                throw std::runtime_error("image need 3 or 4 channels");

            auto format = img.channels == 4 ? GL_RGBA : GL_RGB;

            std::string file_name = split_string(file, '/').back();
            glTexImage2D(
                file_to_tex_id[file_name], 0, GL_RGBA, img.width, img.height, 0, format,
                GL_UNSIGNED_BYTE, img.pixels);
        }

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

        curr_texture += 1;
    }

    return program;
}

/*
 * Program
 */

std::map<std::string, GLenum> Program::get_file_to_texture_id() {
    return {
        {"posx.png", GL_TEXTURE_CUBE_MAP_POSITIVE_X}, {"posy.png", GL_TEXTURE_CUBE_MAP_POSITIVE_Y},
        {"posz.png", GL_TEXTURE_CUBE_MAP_POSITIVE_Z}, {"negx.png", GL_TEXTURE_CUBE_MAP_NEGATIVE_X},
        {"negy.png", GL_TEXTURE_CUBE_MAP_NEGATIVE_Y}, {"negz.png", GL_TEXTURE_CUBE_MAP_NEGATIVE_Z}};
}

Program::~Program() {
    glDeleteShader(vertex_shader_id);
    glDeleteShader(fragment_shader_id);

    for (const auto &buffer_id: buffer_ids | std::views::values) glDeleteBuffers(1, &buffer_id);

    for (const auto &tuple: tex_name_to_idx_id | std::views::values) {
        auto [texture_index, cube_texture_id] = tuple;
        glDeleteTextures(1, &cube_texture_id);
    }

    glDeleteProgram(program_id);
}

void Program::use() const { glUseProgram(program_id); }

template<typename F, class... T>
void Program::uniform_(F uniform_fun, const std::string &name, T... args) {
    uniform_fun(uniform_handles[name], args...);
}

void Program::uniform_mat4(const std::string &name, glm::mat4 mat4) {
    uniform_(glUniformMatrix4fv, name, 1, GL_FALSE, glm::value_ptr(mat4));
}

void Program::uniform_vec4(const std::string &name, glm::vec4 vec4) {
    uniform_(glUniform4fv, name, 1, glm::value_ptr(vec4));
}

void Program::uniform_vec3(const std::string &name, glm::vec3 vec3) {
    uniform_(glUniform3fv, name, 1, glm::value_ptr(vec3));
}

void Program::uniform_float(const std::string &name, const float f) {
    uniform_(glUniform1f, name, f);
}

void Program::attrib(
    const std::string &name, const std::string &buffer_name, const int data_size, const int stride,
    const int offset) {
    glBindBuffer(GL_ARRAY_BUFFER, buffer_ids[buffer_name]);

    glEnableVertexAttribArray(attribute_handles[name]);
    glVertexAttribPointer(
        attribute_handles[name], data_size, GL_FLOAT, GL_FALSE, stride, (char *) nullptr + offset);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Program::disable_attrib_array() {
    for (const auto attrib_id: attribute_handles | std::views::values)
        glDisableVertexAttribArray(attrib_id);
}

void Program::draw_arrays(const GLenum type, const int from, const int nb_vertices) {
    glDrawArrays(type, from, nb_vertices);
}

void Program::texture_(const GLenum texture_target, const std::string &name) {
    auto [texture_index, texture_id] = tex_name_to_idx_id[name];

    glActiveTexture(texture_index);
    glBindTexture(texture_target, texture_id);

    uniform_(glUniform1i, name, texture_index - GL_TEXTURE0);
}

void Program::cube_texture(const std::string &cube_texture_name) {
    texture_(GL_TEXTURE_CUBE_MAP, cube_texture_name);
}

void Program::texture(const std::string &texture_name) { texture_(GL_TEXTURE_2D, texture_name); }

void Program::disable_texture_(const GLenum texture_target) { glBindTexture(texture_target, 0); }

void Program::disable_cube_texture() { disable_texture_(GL_TEXTURE_CUBE_MAP); }

void Program::disable_texture() { disable_texture_(GL_TEXTURE_2D); }
