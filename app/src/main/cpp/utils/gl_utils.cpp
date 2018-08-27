//
// Created by samuel on 27/08/18.
//

#include "gl_utils.h"

void checkGLError() {
	GLenum err;
	bool failed = false;
	while ((err = glGetError()) != GL_NO_ERROR) {
		__android_log_print(ANDROID_LOG_DEBUG, "PhyVR", "OpenGL Error : %d", err);
		failed = true;
	}
	if (failed)
		exit(0);
}