//
// Created by samuel on 18/03/2023.
//

#include "./shader.h"

GLuint load_shader(
    const std::shared_ptr<AbstractFileReader> &text_reader, GLenum type,
    const std::string &file_name) {
    GLuint shader = glCreateShader(type);

    std::string shader_source = text_reader->read_text(file_name);
    const char *c_str = shader_source.c_str();

    glShaderSource(shader, 1, &c_str, nullptr);
    glCompileShader(shader);

    return shader;
}
