//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_VK_ERRORS_H
#define ARENAI_VK_ERRORS_H

#include <string>

#include "./vk.h"

namespace arenai::view {

    // Throws std::runtime_error with the result name when result is not VK_SUCCESS.
    void vk_check(VkResult result, const std::string &message);

}// namespace arenai::view

#endif// ARENAI_VK_ERRORS_H
