#include <GLES2/gl2.h>
#include "shader.h"

GLuint loadShader(GLenum type, const char* shaderSource) {

    GLuint shader = glCreateShader(type);

    glShaderSource( shader, 1, &shaderSource, NULL );

    return shader;
}