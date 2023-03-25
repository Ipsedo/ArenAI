//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_SHADER_H
#define PHYVR_SHADER_H

#include <GLES3/gl3.h>
#include <android/asset_manager.h>
#include <string>

GLuint load_shader(AAssetManager *mgr, GLenum type,
                   const std::string &file_name);

#endif // PHYVR_SHADER_H
