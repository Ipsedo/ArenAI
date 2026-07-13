//
// Created by samuel on 18/03/2023.
//

#include "./shader.h"

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace arenai;

namespace arenai::view {

    GLuint load_shader(const GLenum type, const std::filesystem::path &glsl_relative_path) {
        const GLuint shader = glCreateShader(type);

        const auto shaders_path =
            std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "shaders";

        std::ifstream ifs(shaders_path / glsl_relative_path, std::ios::in);

        if (!ifs.is_open()) throw std::runtime_error(glsl_relative_path);

        std::stringstream buffer;
        buffer << ifs.rdbuf();
        const std::string shader_source = buffer.str();

        const char *c_str = shader_source.c_str();

        glShaderSource(shader, 1, &c_str, nullptr);
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
            throw std::runtime_error(
                "shader compilation failed (" + glsl_relative_path.string() + "): " + info_log);
        }

        return shader;
    }

}// namespace arenai::view
