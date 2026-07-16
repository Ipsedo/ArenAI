//
// Created by samuel on 18/03/2023.
//

#ifndef ARENAI_SHADER_H
#define ARENAI_SHADER_H

#include <filesystem>

#include <GLES3/gl3.h>

namespace arenai::view {

    // Compiles one of the GLSL sources embedded at build time (see
    // cmake/ArenaiEmbedShaders.cmake), keyed by file name e.g. "diffuse_vs.glsl".
    GLuint load_shader(GLenum type, const std::string &glsl_name);

}// namespace arenai::view

#endif// ARENAI_SHADER_H
