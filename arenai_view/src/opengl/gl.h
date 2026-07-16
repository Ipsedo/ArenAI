//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_GL_H
#define ARENAI_GL_H

// Desktop OpenGL 3.3 core entry points. On Linux the GL library (Mesa / NVIDIA,
// dispatched through libglvnd) exports the full modern API as real symbols, so
// declaring the extension prototypes and linking OpenGL::GL is enough — no
// function loader (GLEW/glad/epoxy) is needed.
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#endif// ARENAI_GL_H
