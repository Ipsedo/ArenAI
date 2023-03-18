//
// Created by samuel on 18/03/2023.
//

#include "shader.h"

#include <android/asset_manager.h>

#include "../utils/asset.h"


GLuint load_shader(AAssetManager *mgr, GLenum type, const std::string &file_name) {
    GLuint shader = glCreateShader(type);

    std::string shader_source = read_asset(mgr, file_name);
    const char *c_str = shader_source.c_str();

    glShaderSource(shader, 1, &c_str, nullptr);
    glCompileShader(shader);

    return shader;
}