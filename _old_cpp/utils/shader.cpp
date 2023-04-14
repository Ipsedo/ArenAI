#include "shader.h"
#include <GLES3/gl3.h>
#include <android/log.h>
#include <cstdlib>

GLuint loadShader(GLenum type, const char *shaderSource) {

  GLuint shader = glCreateShader(type);

  glShaderSource(shader, 1, &shaderSource, NULL);
  glCompileShader(shader);

  GLint success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

  if (!success) {
    __android_log_print(ANDROID_LOG_ERROR, "PhyVR", "%s",
                        "shader compilation problem");
    __android_log_print(ANDROID_LOG_DEBUG, "PhyVR", "%s", shaderSource);
    exit(SHADER_ERROR);
  }
  return shader;
}
