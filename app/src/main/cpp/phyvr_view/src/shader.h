//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_SHADER_H
#define PHYVR_SHADER_H

#include <memory>
#include <string>

#include <GLES3/gl3.h>

#include <phyvr_utils/file_reader.h>

GLuint load_shader(
    const std::shared_ptr<AbstractFileReader> &text_reader, GLenum type,
    const std::string &file_name);

#endif// PHYVR_SHADER_H
