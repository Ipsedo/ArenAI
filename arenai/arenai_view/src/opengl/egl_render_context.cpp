//
// Created by samuel on 08/07/2026.
//

#include "./egl_render_context.h"

#include <cstddef>
#include <stdexcept>

namespace arenai::view {

    /*
 * EglRenderContext
 */

    void EglRenderContext::make_current() {
        if (eglMakeCurrent(get_display(), get_surface(), get_surface(), get_context()) != EGL_TRUE)
            throw std::runtime_error("Can't make context");
    }

    void EglRenderContext::release_current() {
        eglMakeCurrent(get_display(), EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }

    /*
 * HeadlessEglContext
 */

    HeadlessEglContext::HeadlessEglContext() {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY) throw std::runtime_error("eglGetDisplay() failed");

        EGLint major, minor;
        if (!eglInitialize(display, &major, &minor))
            throw std::runtime_error("eglInitialize() failed");

        const EGLint cfg_attribs[] = {
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
        if (!eglChooseConfig(display, cfg_attribs, configs, 64, &ncfg) || ncfg < 1)
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
        constexpr EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctx_attribs);
        if (context == EGL_NO_CONTEXT) throw std::runtime_error("eglCreateContext() failed");
    }

    EGLDisplay HeadlessEglContext::get_display() { return display; }

    EGLSurface HeadlessEglContext::get_surface() { return EGL_NO_SURFACE; }

    EGLContext HeadlessEglContext::get_context() { return context; }

    /*
 * NativeEglContext
 */

    NativeEglContext::NativeEglContext(
        const EGLDisplay &display, const EGLSurface &surface, const EGLContext &context)
        : display(display), surface(surface), context(context) {}

    EGLDisplay NativeEglContext::get_display() { return display; }

    EGLSurface NativeEglContext::get_surface() { return surface; }

    EGLContext NativeEglContext::get_context() { return context; }

}// namespace arenai::view
