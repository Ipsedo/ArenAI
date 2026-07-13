//
// Created by samuel on 18/03/2023.
//

#include "./shader.h"

#include <stdexcept>
#include <string>

#include "./shaders_embedded.h"

namespace arenai::view {

    GLuint load_shader(const GLenum type, const std::string &glsl_name) {
        const auto entry = EMBEDDED_SHADERS.find(glsl_name);
        if (entry == EMBEDDED_SHADERS.end())
            throw std::runtime_error("unknown embedded shader: " + glsl_name);

        const std::string_view shader_source = entry->second;

        const GLuint shader = glCreateShader(type);

        const char *c_str = shader_source.data();
        const auto length = static_cast<GLint>(shader_source.size());
        glShaderSource(shader, 1, &c_str, &length);
        glCompileShader(shader);

        GLint compile_status = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
        if (compile_status != GL_TRUE) {
            GLint log_length = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
            std::string info_log(log_length > 0 ? log_length : 1, '\0');
            glGetShaderInfoLog(
                shader, static_cast<GLsizei>(info_log.size()), nullptr, info_log.data());
            glDeleteShader(shader);
            throw std::runtime_error("shader compilation failed (" + glsl_name + "): " + info_log);
        }

        return shader;
    }

}// namespace arenai::view
