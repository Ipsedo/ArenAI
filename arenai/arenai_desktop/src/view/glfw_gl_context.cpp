//
// Created by samuel on 11/03/2026.
//

#include "./glfw_gl_context.h"

#include <csignal>

#include <GLFW/glfw3native.h>

namespace arenai::desktop {

    GlfwGlContext::GlfwGlContext(GLFWwindow *window) : window(window) {
        glfwMakeContextCurrent(window);
    }

    EGLDisplay GlfwGlContext::get_display() { return glfwGetEGLDisplay(); }

    EGLSurface GlfwGlContext::get_surface() { return glfwGetEGLSurface(window); }

    EGLContext GlfwGlContext::get_context() { return glfwGetEGLContext(window); }

}// namespace arenai::desktop
