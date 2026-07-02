//
// Created by samuel on 01/07/2026.
//

#ifndef ARENAI_CORE_TESTS_LOCAL_GL_CONTEXT_H
#define ARENAI_CORE_TESTS_LOCAL_GL_CONTEXT_H

#include <arenai_view/renderer.h>

class LocalGlContext final : public arenai::view::AbstractGLContext {
public:
    LocalGlContext();

    EGLDisplay get_display() override;

    EGLSurface get_surface() override;

    EGLContext get_context() override;

private:
    EGLDisplay display;
    EGLContext context;
};

#endif// ARENAI_CORE_TESTS_LOCAL_GL_CONTEXT_H
