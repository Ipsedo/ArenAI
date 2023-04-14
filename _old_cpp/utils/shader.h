#ifndef PHYVR_SHADER_H
#define PHYVR_SHADER_H

#include <GLES3/gl3.h>

#define SHADER_ERROR 369

GLuint loadShader(GLenum type, const char *shaderSource);

#endif
