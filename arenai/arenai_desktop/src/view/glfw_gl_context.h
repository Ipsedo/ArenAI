//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_DESKTOP_GLFW_GL_CONTEXT_H
#define ARENAI_DESKTOP_GLFW_GL_CONTEXT_H

#include <arenai_view/renderer.h>

class GlfwGlContext : public AbstractGLContext {
public:
    GlfwGlContext(int window_width, int window_height);

    EGLDisplay get_display() override;
    EGLSurface get_surface() override;
    EGLContext get_context() override;
};

#endif//ARENAI_DESKTOP_GLFW_GL_CONTEXT_H
