#ifndef ARENAI_VK_SHADER_MODULES_H
#define ARENAI_VK_SHADER_MODULES_H

#include <string>

#include "./vk.h"

namespace arenai::view {

    VkShaderModule load_shader_module(const VkDevice &device, const std::string &glsl_name);

}// namespace arenai::view

#endif// ARENAI_VK_SHADER_MODULES_H
