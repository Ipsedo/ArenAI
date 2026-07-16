//
// Created by samuel on 19/03/2023.
//

#ifndef ARENAI_ERRORS_H
#define ARENAI_ERRORS_H

#include <string>

namespace arenai::view {

    void check_gl_error(const std::string &message);

}// namespace arenai::view

#endif// ARENAI_ERRORS_H
