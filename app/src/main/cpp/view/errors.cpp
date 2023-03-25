//
// Created by samuel on 19/03/2023.
//

#include "./errors.h"

#include <GLES3/gl3.h>
#include <iostream>
#include <stdexcept>

#include "../utils/logging.h"

void check_gl_error(const std::string &message) {
  GLenum error_code;

  bool has_error;

  while ((error_code = glGetError()) != GL_NO_ERROR) {
    LOG_ERROR("GL_ERROR \"%s\": %d", message.c_str(), error_code);
    has_error = true;
  }

  /*if (has_error) {
      exit(1);
  }*/
}
