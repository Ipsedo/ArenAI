//
// Created by samuel on 26/06/2026.
//

#include <EGL/egl.h>

#include "./local_gl_context.h"

LocalGlContext::LocalGlContext() {
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

    EGLConfig configs[64];
    EGLint ncfg = 0;
    if (!eglChooseConfig(display, cfgAttribs, configs, 64, &ncfg) || ncfg < 1)
        throw std::runtime_error("eglChooseConfig() failed");

    EGLConfig config = nullptr;
    for (EGLint i = 0; i < ncfg; ++i) {
        EGLint r = 0, g = 0, b = 0, a = 0;
        eglGetConfigAttrib(display, configs[i], EGL_RED_SIZE, &r);
        eglGetConfigAttrib(display, configs[i], EGL_GREEN_SIZE, &g);
        eglGetConfigAttrib(display, configs[i], EGL_BLUE_SIZE, &b);
        eglGetConfigAttrib(display, configs[i], EGL_ALPHA_SIZE, &a);
        if (r == 8 && g == 8 && b == 8 && a == 8) {
            config = configs[i];
            break;
        }
    }
    if (config == nullptr) throw std::runtime_error("No 8-bit RGBA EGLConfig available");

    eglBindAPI(EGL_OPENGL_ES_API);
    constexpr EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctxAttribs);
    if (context == EGL_NO_CONTEXT) throw std::runtime_error("eglCreateContext() failed");
}

EGLDisplay LocalGlContext::get_display() { return display; }

EGLSurface LocalGlContext::get_surface() { return std::nullptr_t(); }

EGLContext LocalGlContext::get_context() { return context; }
