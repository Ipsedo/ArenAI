//
// Created by samuel on 12/10/2025.
//

#include "./train_gl_context.h"

#include "arenai_view/errors.h"

TrainGlContext::TrainGlContext() {
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) throw std::runtime_error("eglGetDisplay() failed");
    EGLint major, minor;
    if (!eglInitialize(display, &major, &minor)) throw std::runtime_error("eglInitialize() failed");

    const EGLint cfgAttribs[] = {
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        EGL_DEPTH_SIZE,
        16,
        EGL_STENCIL_SIZE,
        8,
        EGL_SAMPLES,
        0,
        EGL_NONE};
    EGLConfig config;
    EGLint ncfg = 0;
    if (!eglChooseConfig(display, cfgAttribs, &config, 1, &ncfg) || ncfg < 1)
        throw std::runtime_error("eglChooseConfig() failed");

    eglBindAPI(EGL_OPENGL_ES_API);
    constexpr EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctxAttribs);
    if (context == EGL_NO_CONTEXT) throw std::runtime_error("eglCreateContext() failed");
}

EGLDisplay TrainGlContext::get_display() { return display; }

EGLSurface TrainGlContext::get_surface() { return std::nullptr_t(); }

EGLContext TrainGlContext::get_context() { return context; }
