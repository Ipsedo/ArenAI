//
// Created by samuel on 08/07/2026.
//

#ifndef ARENAI_GL_WINDOW_H
#define ARENAI_GL_WINDOW_H

#include <EGL/egl.h>

#include <arenai_view/window.h>

namespace arenai::view {

    class AbstractGlWindow : public AbstractWindow {
    public:
        virtual EGLDisplay egl_display() const = 0;
        virtual EGLSurface egl_surface() const = 0;
        virtual EGLContext egl_context() const = 0;
    };

}// namespace arenai::view

#endif//ARENAI_GL_WINDOW_H
