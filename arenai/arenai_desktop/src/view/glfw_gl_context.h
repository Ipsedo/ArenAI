//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_DESKTOP_GLFW_GL_CONTEXT_H
#define ARENAI_DESKTOP_GLFW_GL_CONTEXT_H

#define GLFW_EXPOSE_NATIVE_EGL
#include <GLFW/glfw3.h>

#include <arenai_view/renderer.h>

class GlfwGlContext : public AbstractGLContext {
public:
    explicit GlfwGlContext(GLFWwindow *window);

    EGLDisplay get_display() override;
    EGLSurface get_surface() override;
    EGLContext get_context() override;

private:
    GLFWwindow *window;
};

#endif//ARENAI_DESKTOP_GLFW_GL_CONTEXT_H
