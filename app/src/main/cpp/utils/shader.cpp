#include <GLES2/gl2.h>
#include "shader.h"

GLuint loadShader(GLenum type, const char* shaderSource) {

    GLuint shader = glCreateShader(type);

    glShaderSource( shader, 1, &shaderSource, NULL );
    glCompileShader(shader);


    /*GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    __android_log_print(ANDROID_LOG_INFO, "POIR", "POIR %s", success == 0 ? "pb" : "good");*/
    return shader;
}