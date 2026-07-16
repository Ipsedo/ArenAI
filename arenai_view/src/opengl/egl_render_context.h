//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_EGL_RENDER_CONTEXT_H
#define ARENAI_EGL_RENDER_CONTEXT_H

#include <EGL/egl.h>

#include <arenai_view/renderer.h>

#include "./gl.h"

namespace arenai::view {

    class EglRenderContext : public AbstractRenderContext {
    public:
        void make_current() override;
        void release_current() override;

        virtual EGLDisplay get_display() = 0;
        virtual EGLSurface get_surface() = 0;
        virtual EGLContext get_context() = 0;

    protected:
        // Desktop GL core profile provides no default vertex array object, so
        // one is bound per context in make_current(). VAOs are container
        // objects and are never shared between contexts.
        GLuint vao_ = 0;
    };

    class HeadlessEglContext final : public EglRenderContext {
    public:
        HeadlessEglContext();

        EGLDisplay get_display() override;
        EGLSurface get_surface() override;
        EGLContext get_context() override;

    private:
        EGLDisplay display;
        EGLContext context;
    };

    class NativeEglContext final : public EglRenderContext {
    public:
        NativeEglContext(
            const EGLDisplay &display, const EGLSurface &surface, const EGLContext &context);

        EGLDisplay get_display() override;
        EGLSurface get_surface() override;
        EGLContext get_context() override;

    private:
        EGLDisplay display;
        EGLSurface surface;
        EGLContext context;
    };

}// namespace arenai::view

#endif// ARENAI_EGL_RENDER_CONTEXT_H
