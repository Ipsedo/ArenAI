//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_SHADER_MODULES_H
#define ARENAI_VK_SHADER_MODULES_H

#include <string>

#include "./vk.h"

namespace arenai::view {

    // Creates a shader module from one of the SPIR-V binaries embedded at
    // build time (see cmake/ArenaiCompileShaders.cmake), keyed by the source
    // file name e.g. "diffuse_vs.glsl". The caller destroys the module once
    // its pipelines are built.
    VkShaderModule load_shader_module(VkDevice device, const std::string &glsl_name);

}// namespace arenai::view

#endif// ARENAI_VK_SHADER_MODULES_H
