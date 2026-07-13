//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_SHADER_H
#define ARENAI_SHADER_H

#include <memory>
#include <string>

#include <GLES3/gl3.h>

#include <arenai_utils/file_reader.h>

namespace arenai::view {

    GLuint load_shader(GLenum type, const std::filesystem::path &glsl_relative_path);

}// namespace arenai::view

#endif// ARENAI_SHADER_H
