//
// Created by samuel on 18/03/2023.
//

#include "./program.h"

#include <utility>
#include <glm/gtc//type_ptr.hpp>
#include <filesystem>

#include "./constants.h"
#include "./shader.h"
#include "../utils/asset.h"
#include "../utils/string_utils.h"
#include "../utils/logging.h"
#include "./errors.h"

/*
 * Buffer builder
 */

Program::Builder::Builder(AAssetManager *mgr, std::string vertex_shader_path,
                          std::string fragment_shader_path) :
        mgr(mgr),
        vertex_shader_path(std::move(vertex_shader_path)),
        fragment_shader_path(std::move(fragment_shader_path)) {

}

Program::Builder::Builder(AAssetManager *mgr, std::string vertex_shader_path,
                          std::string fragment_shader_path,
                          std::vector<std::string> uniforms, std::vector<std::string> attributes,
                          std::map<std::string, std::vector<float>> buffers,
                          std::map<std::string, std::vector<std::filesystem::path>> cube_textures,
                          std::map<std::string, std::filesystem::path> textures) :
        mgr(mgr),
        vertex_shader_path(std::move(vertex_shader_path)),
        fragment_shader_path(std::move(fragment_shader_path)),
        uniforms(std::move(uniforms)),
        attributes(std::move(attributes)),
        buffers(std::move(buffers)),
        cube_textures(std::move(cube_textures)),
        textures(std::move(textures)) {

}

Program::Builder Program::Builder::add_uniform(const std::string &name) {
    uniforms.push_back(name);
    return {mgr, vertex_shader_path, fragment_shader_path, uniforms, attributes, buffers,
            cube_textures, textures};
}

Program::Builder Program::Builder::add_attribute(const std::string &name) {
    attributes.push_back(name);
    return {mgr, vertex_shader_path, fragment_shader_path, uniforms, attributes, buffers,
            cube_textures, textures};
}

Program::Builder
Program::Builder::add_buffer(const std::string &name, const std::vector<float> &data) {
    buffers.insert({name, data});
    return {mgr, vertex_shader_path, fragment_shader_path, uniforms, attributes, buffers,
            cube_textures, textures};
}

Program::Builder Program::Builder::add_cube_texture(const std::string &name,
                                                    const std::string &cube_textures_root_path) {

    std::filesystem::path root_path(cube_textures_root_path);

    std::vector<std::filesystem::path> full_paths;
    for (const auto &[png_file, _]: Program::get_file_to_texture_id())
        full_paths.push_back(root_path / png_file);

    cube_textures.insert({name, full_paths});

    return {mgr, vertex_shader_path, fragment_shader_path, uniforms, attributes, buffers,
            cube_textures, textures};
}

Program::Builder
Program::Builder::add_texture(const std::string &name, const std::string &texture_path) {
    textures.insert({name, std::filesystem::path(texture_path)});
    return {mgr, vertex_shader_path, fragment_shader_path, uniforms, attributes, buffers,
            cube_textures, textures};
}

std::shared_ptr<Program> Program::Builder::build() {
    std::shared_ptr<Program> program = std::make_shared<Program>();

    program->program_id = glCreateProgram();

    program->vertex_shader_id = load_shader(mgr, GL_VERTEX_SHADER, vertex_shader_path);
    program->fragment_shader_id = load_shader(mgr, GL_FRAGMENT_SHADER, fragment_shader_path);

    glAttachShader(program->program_id, program->vertex_shader_id);
    glAttachShader(program->program_id, program->fragment_shader_id);

    glLinkProgram(program->program_id);

    // buffers
    for (const auto &[name, data]: buffers) {
        program->buffer_ids.insert({name, 0});

        glGenBuffers(1, &program->buffer_ids[name]);

        glBindBuffer(GL_ARRAY_BUFFER, program->buffer_ids[name]);
        glBufferData(GL_ARRAY_BUFFER, int(data.size()) * BYTES_PER_FLOAT, &data[0], GL_STATIC_DRAW);

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

        program->uniform_handles.insert({
                                                name, glGetUniformLocation(program->program_id,
                                                                           name.c_str())
                                        });

        glGenTextures(1, &tex_id);
        glActiveTexture(curr_texture);
        glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id);

        program->tex_name_to_idx_id.insert({name, {curr_texture, tex_id}});

        for (const auto &file: files_path) {
            img_rgb img = read_png(mgr, file);

            std::string file_name = split_string(file, '/').back();
            glTexImage2D(
                    file_to_tex_id[file_name],
                    0, GL_RGBA,
                    img.width, img.height,
                    0, GL_RGBA, GL_UNSIGNED_BYTE,
                    img.pixels);

            delete[] img.pixels;
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
            {"posx.png", GL_TEXTURE_CUBE_MAP_POSITIVE_X},
            {"posy.png", GL_TEXTURE_CUBE_MAP_POSITIVE_Y},
            {"posz.png", GL_TEXTURE_CUBE_MAP_POSITIVE_Z},
            {"negx.png", GL_TEXTURE_CUBE_MAP_NEGATIVE_X},
            {"negy.png", GL_TEXTURE_CUBE_MAP_NEGATIVE_Y},
            {"negz.png", GL_TEXTURE_CUBE_MAP_NEGATIVE_Z}
    };
}

Program::~Program() {
    glDeleteShader(vertex_shader_id);
    glDeleteShader(fragment_shader_id);

    for (const auto &[name, buffer_id]: buffer_ids)
        glDeleteBuffers(1, &buffer_id);

    for (const auto &[name, tuple]: tex_name_to_idx_id) {
        auto [texture_index, cube_texture_id] = tuple;
        glDeleteTextures(1, &cube_texture_id);
    }

    glDeleteProgram(program_id);
}

void Program::use() const {
    glUseProgram(program_id);
}

template<typename F, class... T>
void Program::_uniform(F uniform_fun, const std::string &name, T... args) {
    uniform_fun(uniform_handles[name], args...);
}

void Program::uniform_mat4(const std::string &name, glm::mat4 mat4) {
    _uniform(glUniformMatrix4fv, name, 1, GL_FALSE, glm::value_ptr(mat4));
}

void Program::uniform_vec4(const std::string &name, glm::vec4 vec4) {
    _uniform(glUniform4fv, name, 1, glm::value_ptr(vec4));
}

void Program::uniform_vec3(const std::string &name, glm::vec3 vec3) {
    _uniform(glUniform3fv, name, 1, glm::value_ptr(vec3));
}

void Program::uniform_float(const std::string &name, float f) {
    _uniform(glUniform1f, name, f);
}

void
Program::attrib(const std::string &name, const std::string &buffer_name, int data_size, int stride,
                int offset) {
    glBindBuffer(GL_ARRAY_BUFFER, buffer_ids[buffer_name]);

    glEnableVertexAttribArray(attribute_handles[name]);
    glVertexAttribPointer(attribute_handles[name], data_size, GL_FLOAT, GL_FALSE, stride,
                          (char *) nullptr + offset);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Program::disable_attrib_array() {
    for (auto [name, attrib_id]: attribute_handles)
        glDisableVertexAttribArray(attrib_id);
}

void Program::draw_arrays(GLenum type, int from, int nb_vertices) {
    glDrawArrays(type, from, nb_vertices);
}

void Program::_texture(GLenum texture_target, const std::string &name) {
    auto [texture_index, texture_id] = tex_name_to_idx_id[name];

    glActiveTexture(texture_index);
    glBindTexture(texture_target, texture_id);

    _uniform(glUniform1i, name, texture_index - GL_TEXTURE0);
}

void Program::cube_texture(const std::string &cube_texture_name) {
    _texture(GL_TEXTURE_CUBE_MAP, cube_texture_name);
}

void Program::texture(const std::string &texture_name) {
    _texture(GL_TEXTURE_2D, texture_name);
}

void Program::_disable_texture(GLenum texture_target) {
    glBindTexture(texture_target, 0);
}

void Program::disable_cube_texture() {
    _disable_texture(GL_TEXTURE_CUBE_MAP);
}

void Program::disable_texture() {
    _disable_texture(GL_TEXTURE_2D);
}
