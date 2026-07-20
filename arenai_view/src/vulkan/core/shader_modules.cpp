//
// Created by samuel on 17/07/2026.
//

#include "./shader_modules.h"

#include <stdexcept>

#include "./errors.h"
#include "./shaders_spirv.h"

namespace arenai::view {

    VkShaderModule load_shader_module(const VkDevice device, const std::string &glsl_name) {
        const auto entry = EMBEDDED_SPIRV.find(glsl_name);
        if (entry == EMBEDDED_SPIRV.end())
            throw std::runtime_error("unknown embedded shader: " + glsl_name);

        VkShaderModuleCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = entry->second.size_bytes();
        create_info.pCode = entry->second.data();

        VkShaderModule shader_module = VK_NULL_HANDLE;
        vk_check(
            vkCreateShaderModule(device, &create_info, nullptr, &shader_module),
            "vkCreateShaderModule (" + glsl_name + ")");
        return shader_module;
    }

}// namespace arenai::view
