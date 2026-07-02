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

    GLuint load_shader(
        const std::shared_ptr<utils::AbstractFileReader> &text_reader, GLenum type,
        const std::filesystem::path &file_name);

}// namespace arenai::view

#endif// ARENAI_SHADER_H
